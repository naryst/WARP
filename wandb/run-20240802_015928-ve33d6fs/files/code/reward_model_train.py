from transformers import AutoTokenizer, AutoModelForMaskedLM
import datasets
from trl import RewardTrainer, RewardConfig
import os


os.environ['WANDB_PROJECT']='WARP_imdb'

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-cased")
data = datasets.load_from_disk("comment_pairs")


def tokenize_data(sample):
    tokenized_pos = tokenizer(
        sample["positive_comment"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_neg = tokenizer(
        sample["negative_comment"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    sample["input_ids_chosen"] = tokenized_pos["input_ids"]
    sample["attention_mask_chosen"] = tokenized_pos["attention_mask"]
    sample["input_ids_rejected"] = tokenized_neg["input_ids"]
    sample["attention_mask_rejected"] = tokenized_neg["attention_mask"]

    return sample


data = data.map(tokenize_data, batched=True, num_proc=16)
data.set_format("torch")

# Define reward config
reward_config = RewardConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    learning_rate=1e-5,
    do_eval=False,
    report_to='wandb',
    max_length=512,
    remove_unused_columns=False,
    logging_steps=50,
    save_steps=500,
    run_name='reward_model_train'
)

# Initialize the RewardTrainer
trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=data,
    tokenizer=tokenizer,
    compute_metrics=None,  # Add your own metric computation if needed
)

# Train the model
trainer.train()
