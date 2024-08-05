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
from merging import models_interpolate, slerp_models
import wandb
import hydra
from omegaconf import OmegaConf

lt.monkey_patch()


def weights_check(model1, model2):
    main_model_weights = {k: v.clone() for k, v in model1.named_parameters()}
    reference_model_weights = {k: v.clone() for k, v in model2.named_parameters()}

    # Verify that all parameters are identical
    for name, param in main_model_weights.items():
        assert torch.equal(
            param, reference_model_weights[name]
        ), f"Mismatch found in {name}"

    print("Reference model successfully created and verified.")


# create Adam optimizer with a given model params
def opt(model, lr=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def get_random_prompts(dataset, batch_size):
    r = torch.randint(low=0, high=len(dataset), size=(batch_size,))
    return dataset[r]


def get_score(model, tokenizer, response_text):
    inputs = tokenizer(
        response_text, padding="max_length", truncation=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits


def get_kl(current_policy, anchor_policy):
    kl = current_policy - anchor_policy
    # mathematically correct reduction (https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
    kl = kl.sum(dim=-1) / current_policy.size(0)
    kl = kl.unsqueeze(-1)
    return kl


def reward_KL_penalty(reward, beta, current_policy, anchor_policy):
    kl = get_kl(current_policy, anchor_policy)
    reward -= beta * kl
    return reward, kl


def model_generate_output(prompt, model, tokenizer, generation_config):
    if len(prompt.shape) == 1:
        prompt = prompt.unsqueeze(0)
    result = model.generate(
        prompt,
        return_dict_in_generate=True,
        output_scores=True,
        generation_config=generation_config,
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
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def get_policies(
    model,
    ref_model,
    tokenizer,
    prompt,  # B x L x |V| (B - batchsize, L - prompt len, |V| - vocab size)
    generation_config,
):
    """
    Generate text and compute log probabilities for both the current model and a reference model.

    This function performs the following steps:
    1. Generates text using the current model based on the given prompt.
    2. Computes logits and log probabilities for the generated text using both the current model and a reference model.

    Args:
        model: The current language model used for text generation.
        ref_model: The reference language model used for comparison.
        tokenizer: The tokenizer for the current model.
        prompt (torch.Tensor): The input prompt tensor of shape (B, L, |V|), where B is the batch size,
                               L is the prompt length, and |V| is the vocabulary size.

    Returns:
        tuple: A tuple containing:
            - logprobs (torch.Tensor): Log probabilities of the generated tokens for the current model.
            - anchor_logprobs (torch.Tensor): Log probabilities of the generated tokens for the reference model.
            - full_text (str): The full generated text including the prompt.
    """
    TEMPERATURE = generation_config.temperature
    # ============= STEP 1 - GET POLICIES =============
    # get generation logits for the current model
    full_text_tokens, full_text, cur_logits = model_generate_output(
        prompt, model, tokenizer, generation_config
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

    # get cur model logits in the same way (trying to fix KL)
    cur_logits = model_forward(  # B x (L + L_g) x |V|
        model, query_responses=full_text_tokens, pad_token_id=tokenizer.pad_token_id
    ).logits
    cur_logits = cur_logits[:, prompt_length - 1 : -1]  # B x L_g x |V|

    # assert torch.equal(anchor_logits, cur_logits)
    # weights_check(model, ref_model)

    cur_logits /= TEMPERATURE
    cur_logits_dist = F.log_softmax(cur_logits, dim=-1)
    logprobs = torch.gather(cur_logits_dist, 2, generated.unsqueeze(-1)).squeeze(
        -1
    )  # B x L_g

    anchor_logits /= TEMPERATURE  # set the same temperature as for the generation
    anchor_logits_dist = F.log_softmax(anchor_logits, dim=-1)
    anchor_logprobs = torch.gather(
        anchor_logits_dist, 2, generated.unsqueeze(-1)
    ).squeeze(-1)  # B x L_g

    # assert torch.equal(logprobs, anchor_logprobs)
    return logprobs, anchor_logprobs, full_text


def policy_gradient(
    reward_model,
    model,
    ref_model,
    tokenizer,
    reward_model_tokenizer,
    prompt,  # B x L x |V| (B - batchsize, L - prompt len, |V| - vocab size)
    optimizer,
    beta,
    generation_config,
    make_optimizer_step=True,
):
    logprobs, anchor_logprobs, full_text = get_policies(
        model, ref_model, tokenizer, prompt, generation_config
    )

    # ============= STEP 2 - KL PENALIZED REWARD =============
    reward = get_score(reward_model, reward_model_tokenizer, full_text)
    reward_kl_penalized, kl = reward_KL_penalty(reward, beta, logprobs, anchor_logprobs)

    # ============= STEP 3 - POLICY GRADIENT =============
    loss = -(reward * logprobs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return reward, kl, loss


def KL_reward_paretto_front(
    model_sft,  # Pre-trained model (student fine-tuned)
    model_slerp,  # Pre-trained model for slerp interpolation
    batch,  # Batch of input data
    tokenizer,  # Tokenizer for the models
    reward_model,  # Model used for computing reward score
    reward_model_tokenizer,  # Tokenizer for the reward model
    generation_config,  # Configuration for text generation
):
    # List of interpolation coefficients
    nus = [0, 0.1, 0.3, 0.5, 0.8, 1]
    points = []
    for nu in nus:
        sft_copy = deepcopy(model_sft)
        # Interpolate between the model_slerp and the deep copy of model_sft
        merged_model = models_interpolate(model_slerp, sft_copy, mu=nu)

        # Generate policies
        merged_policy, sft_policy, response = get_policies(
            merged_model, model_sft, tokenizer, batch, generation_config
        )

        KL = get_kl(merged_policy, sft_policy).mean().item()
        reward = get_score(reward_model, reward_model_tokenizer, response).mean().item()
        points.append((KL, reward))
    return points


def step_log(step, reward, kl, loss, track):
    info = {
        "step": step,
        "reward": reward.mean().item(),
        "kl": kl.mean().item(),
        "loss": loss.item(),
    }
    if track:
        wandb.log(info, step=step)
    else:
        print(info)


def final_log(points, train_kls, train_rewards):
    wandb.log(
        {
            "KL_vs_Reward": wandb.plot.line(
                wandb.Table(
                    data=[[kl, reward] for kl, reward in points],
                    columns=["KL", "Reward"],
                ),
                "KL",
                "Reward",
                title="KL-Reward Pareto Front",
            )
        }
    )

    wandb.log(
        {
            "train KL_vs_Reward": wandb.plot.line(
                wandb.Table(
                    data=[[kl, reward] for kl, reward in zip(train_kls, train_rewards)],
                    columns=["KL", "Reward"],
                ),
                "KL",
                "Reward",
                title="train KL-Reward",
            )
        }
    )


def warp(
    model_init,  # Initial model to be optimized
    ref_model,  # Reference model for computing reward and KL divergence
    tokenizer,  # Tokenizer for the models
    reward_model,  # Model used for computing reward score
    reward_model_tokenizer,  # Tokenizer for the reward model
    X,  # Dataset for training
    opt,  # Optimizer function
    cfg,  # Configuration object containing various hyperparameters and settings
):
    train_rewards = []
    train_kls = []
    generation_config = GenerationConfig(**cfg.generation_config)

    # Main loop for the number of WARP iterations
    for i in range(cfg.warp.I):
        rl_runs = []
        for m in range(cfg.warp.M):
            model_m = deepcopy(model_init)
            model_ema_m = deepcopy(model_init)
            optimizer = opt(model_m, lr=cfg.training.lr)

            for t in range(cfg.warp.T):
                prompt = get_random_prompts(X, cfg.training.batch_size)

                # Perform a policy gradient update and get reward, KL divergence, and loss
                reward, kl, loss = policy_gradient(
                    reward_model,
                    model_m,
                    model_ema_m,
                    tokenizer,
                    reward_model_tokenizer,
                    prompt,
                    optimizer,
                    cfg.warp.beta,
                    generation_config,
                )

                step = (
                    t + (cfg.warp.T * m) + (cfg.warp.M * cfg.warp.T * i)
                )  # Calculate the current step

                if step % cfg.wandb.log_steps == 0:
                    # Recalculate reward and KL with the reference model instead of EMA model
                    point_reward, point_kl, point_loss = policy_gradient(
                        reward_model,
                        model_m,
                        ref_model,
                        tokenizer,
                        reward_model_tokenizer,
                        prompt,
                        optimizer,
                        cfg.warp.beta,
                        generation_config,
                        make_optimizer_step=False,  # policies just for metrics, don't optimize
                    )
                    step_log(step, point_reward, point_kl, point_loss, cfg.wandb.track)

                    train_rewards.append(point_reward.mean().item())
                    train_kls.append(point_kl.mean().item())

                # Update the EMA model by interpolating with the current model
                model_ema_m = models_interpolate(model_m, model_ema_m, mu=cfg.warp.mu)

            # Store the final EMA model for the current run
            rl_runs.append(model_ema_m)

        assert len(rl_runs) == 2, NotImplementedError

        # Perform spherical linear interpolation (slerp) between the initial model and the two EMA models
        model_slerp = slerp_models(
            model_init, rl_runs[0], rl_runs[1], Lambda=1 / cfg.warp.M
        )

        # Update the initial model by interpolating with the slerp model
        model_init = models_interpolate(model_slerp, model_init, mu=cfg.warp.nu)

    test_batch = get_random_prompts(X, cfg.training.batch_size)
    # Compute the Pareto front of KL divergence and reward
    points = KL_reward_paretto_front(
        ref_model,
        model_init,
        test_batch,
        tokenizer,
        reward_model,
        reward_model_tokenizer,
        generation_config,
    )

    if cfg.wandb.track:
        final_log(points, train_kls, train_rewards)

    wandb.finish()

    return points, model_init


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg):
    device = torch.device(cfg.training.device)
    if device == torch.device("cuda"):
        device = torch.device(f"cuda:{cfg.training.device_id}")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb.track:
        run = wandb.init(  # noqa: F841
            project=cfg.wandb.project_name, name=cfg.wandb.run_name, config=cfg_dict
        )
    generation_config = GenerationConfig(**cfg.generation_config)

    prompts = datasets.load_from_disk(cfg.data.data_path)
    model_checkpoint = cfg.model.generator_checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config = generation_config
    ref_model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
    prompts_ids = prompts["prompt_input_ids"].to(device)

    reward_model_checkpoint = cfg.model.reward_model_checkpoint
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_checkpoint, ignore_mismatched_sizes=True, num_labels=1
    ).to(device)
    reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_checkpoint)
    KL_reward_paretto_front, trained_model = warp(
        model,
        ref_model,
        tokenizer,
        reward_model,
        reward_model_tokenizer,
        prompts_ids,
        opt,
        cfg,
    )
    trained_model.save_pretrained(cfg.model.save_path)
    tokenizer.save_pretrained(cfg.model.save_path)


if __name__ == "__main__":
    main()