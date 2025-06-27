import os
import math
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import load_from_disk


# Load dataset
data_path = "ml/data/processed_dataset"
dataset = load_from_disk(data_path)
train_dataset = dataset.shuffle(seed=42).select(range(int(0.9 * len(dataset))))
eval_dataset = dataset.shuffle(seed=42).select(range(int(0.1 * len(dataset))))

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="no",        # We save manually
    evaluation_strategy="no",  # Safe: not supported in some versions
    report_to="none"
)

# Custom callback to save every 10% of total steps
class SaveEvery10PercentCallback(TrainerCallback):
    def __init__(self, total_steps, output_base="checkpoints/"):
        self.total_steps = total_steps
        self.checkpoint_interval = max(1, math.floor(total_steps / 10))  # 10%
        self.output_base = output_base

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.checkpoint_interval == 0:
            save_path = os.path.join(self.output_base, f"checkpoint-{state.global_step}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"[✓] Saved checkpoint at step {state.global_step} to {save_path}")

# Calculate total training steps
steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
total_steps = steps_per_epoch * training_args.num_train_epochs

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[SaveEvery10PercentCallback(total_steps)]
)

# Train
trainer.train()
