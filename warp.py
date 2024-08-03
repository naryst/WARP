import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoModelForSequenceClassification,
)
import lovely_tensors as lt
import torch.nn.functional as F
from copy import deepcopy


lt.monkey_patch()
TEMPERATURE = 0.7
device = torch.device("cuda")

prompts = datasets.load_from_disk("prompts_dataset_tokenized")
model_checkpoint = "lvwerra/gpt2-imdb"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token_id = tokenizer.eos_token_id

generation_config = GenerationConfig(
    bos_token_id=50256,
    eos_token_id=50256,
    pad_token_id=50256,
    max_length=512,
    use_cache=True,  # сохранять KV для ускорения генерации
    # from ppov2_trainer.py source code
    temperature=TEMPERATURE,
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
    return_tensors="pt",
)
tokenizer_config = {
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt",
}
model.generation_config = generation_config
ref_model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)


def weights_check(model1, model2):
    main_model_weights = {k: v.clone() for k, v in model1.named_parameters()}
    reference_model_weights = {k: v.clone() for k, v in model2.named_parameters()}

    # Verify that all parameters are identical
    for name, param in main_model_weights.items():
        assert torch.equal(
            param, reference_model_weights[name]
        ), f"Mismatch found in {name}"

    print("Reference model successfully created and verified.")

reward_model_checkpoint = "reward_model/final_checkpoint"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_checkpoint, ignore_mismatched_sizes=True, num_labels=1
).to(device)
reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_checkpoint)


def get_score(model, tokenizer, response_text):
    inputs = tokenizer(response_text, **tokenizer_config).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits


def reward_KL_penalty(reward, beta, current_policy, anchor_policy):
    kl = F.kl_div(
        current_policy, anchor_policy.exp(), reduction="none", log_target=True
    )
    # mathematically correct reduction (https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
    kl = kl.sum(dim=-1) / current_policy.size(0)
    kl = kl.unsqueeze(-1)
    reward -= beta * kl
    return reward, kl

def model_generate_output(prompt, model, tokenizer):
    if len(prompt.shape) == 1:
        prompt = prompt.unsqueeze(0)
    result = model.generate(
        prompt,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated = result["sequences"]
    scores = result["scores"]
    scores = torch.stack(scores, 1)
    decoded_result = tokenizer.batch_decode(generated)
    return generated, decoded_result, scores

def model_forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    print(input_ids)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def policy_gradient(
    model,
    ref_model,
    tokenizer,
    prompt,  # B x L x |V| (B - batchsize, L - prompt len, |V| - vocab size)
    optimizer,
    beta,
):
    # ============= STEP 1 - GET POLICIES =============
    # get generation logits for the current model
    full_text_tokens, full_text, cur_logits = model_generate_output(
        prompt, model, tokenizer
    )
    # full_text_tokens - B x (L + L_g) (L_g is num of generated tokens)
    # cur_Logits - B x L_g x |V|

    prompt_length = prompt.size(1)
    generated = full_text_tokens[:, prompt_length:]  # B x L_g

    # get anchor model logits for the same tokens, as generated
    anchor_logits = model_forward(  # B x (L + L_g) x |V|
        ref_model, query_responses=full_text_tokens, pad_token_id=tokenizer.pad_token_id
    ).logits
    anchor_logits = anchor_logits[:, prompt_length - 1 : -1]  # B x L_g x |V|

    cur_logits_dist = F.log_softmax(cur_logits, dim=-1)
    logprobs = torch.gather(cur_logits_dist, 2, generated.unsqueeze(-1)).squeeze(
        -1
    )  # B x L_g

    anchor_logits /= TEMPERATURE # set the same temperature as for the generation
    anchor_logits_dist = F.log_softmax(anchor_logits, dim=-1)
    anchor_logprobs = torch.gather(
        anchor_logits_dist, 2, generated.unsqueeze(-1)
    ).squeeze(-1)  # B x L_g

    # ============= STEP 2 - KL PENALIZED REWARD =============
    reward = get_score(reward_model, reward_model_tokenizer, full_text)
    reward, kl = reward_KL_penalty(reward, beta, logprobs, anchor_logprobs)

    # ============= STEP 3 - POLICY GRADIENT =============
    cur_policy = model_forward(  # B x (L + L_g) x |V|
        model, query_responses=full_text_tokens, pad_token_id=tokenizer.pad_token_id
    ).logits
    cur_policy = cur_policy[:, prompt_length - 1 : -1]  # B x L_g x |V|
    log_cur_policy = F.log_softmax(cur_policy, dim=-1)
    action_log_probs = torch.gather(log_cur_policy, 2, generated.unsqueeze(-1)).squeeze(
        -1
    )  # B x L_g
    loss = -(reward * action_log_probs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return reward, kl


def scale_model_weights(model, scaling_factor):
    scaled_model = deepcopy(model)
    with torch.no_grad():
        for param in scaled_model.parameters():
            param *= scaling_factor
    return scaled_model


def sum_model_weights(model1, model2):
    scaled_model = deepcopy(model1)
    with torch.no_grad():
        for param1, param2 in zip(scaled_model.parameters(), model2.parameters()):
            param1 += param2
    return scaled_model


def models_interpolate(model, additional_model, mu):
    scaled_model = scale_model_weights(model, mu)
    scaled_model_ema = scale_model_weights(additional_model, (1 - mu))
    return sum_model_weights(scaled_model, scaled_model_ema)


def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    return (1 - t) * v0 + t * v1


def normalize(v: torch.Tensor, eps: float):
    norm_v = torch.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v


def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t: Float value between 0.0 and 1.0
        v0: Starting layer weights
        v1: Final layer weights
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2: Interpolation vector between v0 and v1
    """
    # Copy the vectors to reuse them later
    v0_copy = v0.detach().clone()
    v1_copy = v1.detach().clone()

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = torch.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if torch.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return res

    # Calculate initial angle between v0 and v1
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return res


def slerp_models(model1, model2, Lambda=0.5):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1 = slerp(Lambda, param1, param2)
    return model1


# create Adam optimizer with a given model params
def opt(model, lr=1e-6):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def get_random_prompts(dataset, batch_size):
    r = torch.randint(low=0, high=len(dataset), size=(batch_size,))
    return dataset[r]


def warp(
    model_init,
    ref_model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    X,
    opt,
    I,
    M,
    T,
    mu,
    nu,
    beta,
    batch_size,
):
    for i in range(I):
        rl_runs = []
        for m in range(M):
            model_m = deepcopy(model_init)
            model_ema_m = deepcopy(model_init)
            for t in range(T):
                optimizer = opt(model_m)
                prompt = get_random_prompts(X, batch_size)
                reward, kl = policy_gradient(
                    model_m, ref_model, tokenizer, prompt, optimizer, beta
                )
                print("STEP:", t)
                print("REWARD:", reward.mean().item(), "|", "KL:", kl.mean().item())
                print("=" * 100)
                model_ema_m = models_interpolate(model_ema_m, model_m, mu=mu)
            rl_runs.append(model_ema_m)
        model_slerp = deepcopy(model_init)
        for model in rl_runs:
            model_slerp = slerp_models(model_slerp, model, Lambda=1 / M)
        model_init = models_interpolate(model_init, model_slerp, mu=nu)
    # return models_interpolate()


prompts_ids = prompts["prompt_input_ids"].to(device)
warp(
    model,
    ref_model,
    tokenizer,
    reward_model,
    reward_model_tokenizer,
    prompts_ids,
    opt,
    I=1,
    M=2,
    T=5000,
    mu=0.001,
    nu=0.3,
    beta=0.1,
    batch_size=16,
)
