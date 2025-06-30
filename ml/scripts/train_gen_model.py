import os
from datasets import load_from_disk
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Load dataset
data_path = "ml/data/gen_dataset"
dataset = load_from_disk(data_path)
train_dataset = dataset.shuffle(seed=42).select(range(int(0.9 * len(dataset))))
eval_dataset = dataset.shuffle(seed=42).select(range(int(0.1 * len(dataset))))

# Load model and tokenizer
model_name = "google/flan-t5-small"  # You can change to flan-t5-base or larger if GPU allows
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing function
def preprocess(example):
    prompt = f"{example['instruction']}\nInput: {example['input']}\nQuestion: {example['question']}"
    model_input = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    label = tokenizer(example["answer"], padding="max_length", truncation=True, max_length=128)
    model_input["labels"] = label["input_ids"]
    return model_input

train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="ml/gen_checkpoints/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="ml/gen_logs/",
    logging_steps=10,
    report_to="none",
    predict_with_generate=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 🚀 Train the model
if __name__ == "__main__":
    trainer.train()
