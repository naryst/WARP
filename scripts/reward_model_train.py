from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
from trl import RewardTrainer, RewardConfig
import torch
import os


os.environ['WANDB_PROJECT']='WARP_imdb'

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased")

#поменять классифаер на предсказание скаляра (в этом случае - reward)
model.classifier = torch.nn.Linear(model.pre_classifier.weight.size(1), 1)
data = datasets.load_from_disk("../comment_pairs")


def tokenize_data(sample):
    """
    # Reward trainer Должен иметь следующие фичи на входе 
    * `input_ids_chosen`
    * `attention_mask_chosen`
    * `input_ids_rejected` 
    * `attention_mask_rejected`

    **Chosen** в нашем случае - позитивный комментарий, **Rejected** - негативный
    """
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

reward_config = RewardConfig(
    output_dir="reward_model",
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
    # save_steps=500,
    run_name='reward_model_train'
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=data,
    tokenizer=tokenizer,
    compute_metrics=None,
)

# Train the model
trainer.train()
trainer.save_model('../reward_model/final_checkpoint')
