import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)
from warp import get_policies, get_score, get_kl
import hydra
import lovely_tensors


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg):
    data = datasets.load_from_disk("prompts_dataset_tokenized_test")
    SAMPLES = 100
    device = cfg.training.device
    rand_inds = torch.randint(low=0, high=len(data), size=(SAMPLES,))
    test_prompts = data["prompt_input_ids"][rand_inds].to(device)

    gen_conf = GenerationConfig(**cfg.generation_config)

    model = AutoModelForCausalLM.from_pretrained("trained_warp_model").to(device)
    sft_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb").to(device)
    tokenizer = AutoTokenizer.from_pretrained("trained_warp_model")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "reward_model/final_checkpoint",
        ignore_mismatched_sizes=True,
        num_labels=1,
    ).to(device)
    reward_model_tokenizer = AutoTokenizer.from_pretrained(
        "reward_model/final_checkpoint"
    )

    logprobs, sft_logprobs, generated_text = get_policies(
        model, sft_model, tokenizer, test_prompts, gen_conf
    )

    logprobs_sft1, logprobs_sft2, generated_text_sft = get_policies(
        sft_model, sft_model, tokenizer, test_prompts, gen_conf
    )

    rewards = get_score(reward_model, reward_model_tokenizer, generated_text)
    kl = get_kl(logprobs, sft_logprobs)
    print(rewards)
    print(kl)
    print('='*100)
    rewards_sft =  get_score(reward_model, reward_model_tokenizer, generated_text_sft)
    kl_sft = get_kl(logprobs_sft1, logprobs_sft2)
    print(rewards_sft)
    print(kl_sft)


if __name__ == "__main__":
    lovely_tensors.monkey_patch()
    main()
