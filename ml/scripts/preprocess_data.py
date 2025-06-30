import json
import os
import re
from datasets import Dataset

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' "\'\n\t')
    return text

def preprocess_for_generation(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    examples = []

    for title, content in articles.items():
        context = clean_text(content)
        question = f"What is {title}?"
        answer = context.split(".")[0].strip() + "."  # Naive answer = first sentence

        examples.append({
            "instruction": "Answer the question based on the text below.",
            "input": context,
            "question": question,
            "answer": answer
        })

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(output_path)
    print(f"✅ Saved {len(examples)} examples to {output_path}")

if __name__ == "__main__":
    preprocess_for_generation(
        input_path="ml/data/wikipedia_dataset.json",
        output_path="ml/data/gen_dataset"
    )
