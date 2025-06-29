# This script creates an evaluation dataset from a portion of the training dataset.
import os
from datasets import load_from_disk

def create_eval_split(input_dataset_path="ml/data/processed_dataset", 
                      output_path="ml/data/eval_dataset", 
                      eval_ratio=0.1):
    """
    Splits a portion of the training dataset and saves it as evaluation dataset.
    """
    dataset = load_from_disk(input_dataset_path)
    dataset = dataset.shuffle(seed=42)

    eval_size = int(len(dataset) * eval_ratio)
    eval_dataset = dataset.select(range(eval_size))

    os.makedirs(output_path, exist_ok=True)
    eval_dataset.save_to_disk(output_path)
    print(f"✅ Saved evaluation dataset with {len(eval_dataset)} samples to '{output_path}'")

if __name__ == "__main__":
    create_eval_split()
