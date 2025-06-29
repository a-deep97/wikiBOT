import os
import torch
import collections
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_from_disk, load_metric

# Load model and tokenizer
model_dir = "ml/checkpoints/checkpoint-82"
model = BertForQuestionAnswering.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)

# Load dataset (processed evaluation set)
eval_dataset_path = "ml/data/processed_dataset"
dataset = load_from_disk(eval_dataset_path)
eval_dataset = dataset.shuffle(seed=42).select(range(int(0.1 * len(dataset))))  # Sample 10% for evaluation

# Metric
metric = load_metric("squad")

def compute_metrics(pred):
    predictions = pred.predictions
    start_logits, end_logits = predictions

    predictions_list = []

    for i, example in enumerate(eval_dataset):
        start = torch.tensor(start_logits[i])
        end = torch.tensor(end_logits[i])

        start_idx = torch.argmax(start).item()
        end_idx = torch.argmax(end).item()

        input_ids = example['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx+1])
        answer = tokenizer.convert_tokens_to_string(tokens)

        predictions_list.append({
            'id': str(i),
            'prediction_text': answer
        })

    references = [{
        'id': str(i),
        'answers': {
            'answer_start': ex['answers']['answer_start'],
            'text': ex['answers']['text']
        }
    } for i, ex in enumerate(eval_dataset)]

    return metric.compute(predictions=predictions_list, references=references)

# Training arguments for evaluation
args = TrainingArguments(
    output_dir="ml/checkpoints/eval",
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True,
    logging_dir="logs",
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Run evaluation
print("\n📊 Running evaluation with EM/F1 scoring...")
results = trainer.evaluate()

# Display results
print("\n📊 Evaluation Metrics:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# Save results to file
output_file = os.path.join(args.output_dir, "evaluation_result.txt")
os.makedirs(args.output_dir, exist_ok=True)
with open(output_file, "w") as f:
    for key, value in results.items():
        f.write(f"{key}: {value:.4f}\n")

print(f"\n✅ Evaluation results saved to {output_file}")
