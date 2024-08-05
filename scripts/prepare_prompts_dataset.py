import datasets
from transformers import AutoTokenizer
import torch
import argparse


def tokenize_sample(sample, prompt_tokens_count=7):
    tokenized_prompt = tokenizer(
        sample["text"], padding="max_length", truncation=True, return_tensors="pt"
    )
    tokenized_prompt["input_ids"] = tokenized_prompt["input_ids"][
        :, :prompt_tokens_count
    ]
    tokenized_prompt["attention_mask"] = tokenized_prompt["attention_mask"][
        :, :prompt_tokens_count
    ]
    ones_tensor = torch.ones_like(tokenized_prompt["attention_mask"], dtype=torch.int64)
    assert torch.equal(
        tokenized_prompt["attention_mask"], ones_tensor
    )  # проверка, что все промпты имеют равное количество токенов
    sample["prompt_input_ids"] = tokenized_prompt["input_ids"]
    sample["prompt_attention_mask"] = tokenized_prompt["attention_mask"]
    return sample


def main(split):
    data = datasets.load_dataset("stanfordnlp/imdb")[split]
    prompt_tokens_count = 10 if split == "train" else 7
    data = data.map(
        tokenize_sample,
        num_proc=16,
        batched=True,
        batch_size=64,
        fn_kwargs={"prompt_tokens_count": prompt_tokens_count},
    )
    data.set_format("torch")
    data.save_to_disk(f"prompts_dataset_tokenized_{split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IMDb data.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (train, test, validation)",
    )
    args = parser.parse_args()
    # Так как промпты составляются для генератора, то берем его токенайзер
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    tokenizer.pad_token_id = tokenizer.unk_token_id  # Изначально в токенайзере нет pad токена. Можно поставить любое значение, так как все равно потом от него избавимся
    main(args.split)
