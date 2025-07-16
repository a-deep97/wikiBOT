import json
import os
import re
from datasets import Dataset

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' "\'\n\t')

def preprocess_for_generation(input_dir, output_path):
    examples = []

    for file in sorted(os.listdir(input_dir)):
        if file.startswith("wikipedia_dataset_") and file.endswith(".json"):
            file_path = os.path.join(input_dir, file)
            print(f"🔍 Processing {file_path}...")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    articles = json.load(f)
            except json.JSONDecodeError:
                print(f"[!] Skipping corrupted file: {file_path}")
                continue

            for title, content in articles.items():
                context = clean_text(content)
                question = f"What is {title}?"
                answer = context.split(".")[0].strip() + "."

                examples.append({
                    "instruction": "Answer the question based on the text below.",
                    "input": context,
                    "question": question,
                    "answer": answer
                })

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(output_path)
    print(f"\nSaved {len(examples)} examples to {output_path}")

if __name__ == "__main__":
    preprocess_for_generation(
        input_dir="ml/data/training_data",
        output_path="ml/data/gen_dataset"
)
