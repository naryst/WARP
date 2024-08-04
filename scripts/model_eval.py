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
import wandb
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg):
    data = datasets.load_from_disk(cfg.data.test_data_path)
    SAMPLES = 100
    device = cfg.training.device
    rand_inds = torch.randint(low=0, high=len(data), size=(SAMPLES,))
    test_prompts = data["prompt_input_ids"][rand_inds].to(device)

    gen_conf = GenerationConfig(**cfg.generation_config)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.save_path).to(device)
    sft_model = AutoModelForCausalLM.from_pretrained(cfg.model.generator_checkpoint).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.generator_checkpoint)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.reward_model_checkpoint,
        ignore_mismatched_sizes=True,
        num_labels=1,
    ).to(device)
    reward_model_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.reward_model_checkpoint
    )

    logprobs, sft_logprobs, generated_text = get_policies(
        model, sft_model, tokenizer, test_prompts, gen_conf
    )

    logprobs_sft1, logprobs_sft2, generated_text_sft = get_policies(
        sft_model, sft_model, tokenizer, test_prompts, gen_conf
    )

    rewards = get_score(reward_model, reward_model_tokenizer, generated_text)
    kl = get_kl(logprobs, sft_logprobs)
    rewards_sft = get_score(reward_model, reward_model_tokenizer, generated_text_sft)
    kl_sft = get_kl(logprobs_sft1, logprobs_sft2)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb.track:
        run = wandb.init(  # noqa: F841
            project=cfg.wandb.project_name, name=cfg.wandb.run_name, config=cfg_dict
        )
        rewards = [[r.item()] for r in rewards]
        rewards_sft = [[r.item()] for r in rewards_sft]
        r_table = wandb.Table(data=rewards, columns=["scores"])
        r_table_sft = wandb.Table(data=rewards_sft, columns=["scores"])

        kl = [[r.item()] for r in kl]
        kl_table = wandb.Table(data=kl, columns=["scores"])

        wandb.log(
            {
                "reward_dist": wandb.plot.histogram(
                    r_table, "scores", title="Rewards distribution"
                )
            }
        )

        wandb.log(
            {
                "reward_dist_sft": wandb.plot.histogram(
                    r_table_sft, "scores", title="Rewards distribution SFT"
                )
            }
        )

        wandb.log(
            {
                "kl_dist": wandb.plot.histogram(
                    kl_table, "scores", title="KL distribution"
                )
            }
        )

    if not cfg.wandb.track:
        print(rewards)
        print(kl)
        print("=" * 100)
        print(rewards_sft)
        print(kl_sft)


if __name__ == "__main__":
    lovely_tensors.monkey_patch()
    main()
