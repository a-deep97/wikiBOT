import json
import re
import os
from datasets import Dataset
from transformers import BertTokenizerFast


def clean_text(text):
    """Remove citations, normalize whitespace, and clean punctuation."""
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations like [1]
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\s([?.!,:;])', r'\1', text)  # Fix spacing before punctuation
    text = text.strip(' "\'\n\t')  # Strip leading/trailing junk
    return text


def chunk_text_token_based(text, tokenizer, max_tokens=512):
    """Split text into chunks by token count using tokenizer offsets."""
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]

    chunks = []
    start_idx = 0

    while start_idx < len(input_ids):
        end_idx = min(start_idx + max_tokens, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunk_text = chunk_text.replace("##", "").strip()
        if chunk_text:
            chunks.append(chunk_text)
        start_idx = end_idx

    return chunks


def extract_answer(chunk):
    """Pick a basic answer span (first sentence or fallback)."""
    chunk = chunk.replace("##", "").strip()
    match = re.match(r'(.+?[.!?])\s', chunk)
    if match:
        answer = match.group(1).strip()
    else:
        answer = chunk[:min(100, len(chunk))].strip()

    answer_start = chunk.find(answer)
    return answer, answer_start


def preprocess_articles(input_path, output_dir, batch_size=100):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    all_examples = []
    temp_batch = []

    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "dataset_incremental.json")

    if os.path.exists(output_json):
        os.remove(output_json)  # start fresh

    for idx, (title, content) in enumerate(articles.items(), 1):
        clean = clean_text(content)
        chunks = chunk_text_token_based(clean, tokenizer)

        for chunk in chunks:
            if "##" in chunk:
                continue

            question = f"What is {title}?"
            answer_text, answer_start = extract_answer(chunk)
            if answer_start == -1 or not answer_text:
                continue

            # Tokenize with offset mappings
            encoded = tokenizer(
                question,
                chunk,
                truncation="only_second",
                padding="max_length",
                max_length=512,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            offsets = encoded["offset_mapping"][0]
            sequence_ids = encoded.sequence_ids()
            start_char = answer_start
            end_char = start_char + len(answer_text)

            start_token = end_token = None
            for i, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
                if seq_id != 1:
                    continue
                if offset[0] <= start_char < offset[1]:
                    start_token = i
                if offset[0] < end_char <= offset[1]:
                    end_token = i
                    break

            if start_token is None or end_token is None:
                continue

            example = {
                "context": chunk,
                "question": question,
                "answers": {
                    "text": [answer_text],
                    "answer_start": [answer_start]
                },
                "title": title,
                "input_ids": encoded["input_ids"][0].tolist(),
                "token_type_ids": encoded["token_type_ids"][0].tolist(),
                "attention_mask": encoded["attention_mask"][0].tolist(),
                "start_positions": start_token,
                "end_positions": end_token
            }

            temp_batch.append(example)
            all_examples.append(example)

        # Save every `batch_size` articles
        if idx % batch_size == 0:
            with open(output_json, 'a', encoding='utf-8') as f:
                for ex in temp_batch:
                    json.dump(ex, f, ensure_ascii=False)
                    f.write("\n")
            print(f"[✓] Processed {idx} articles, saved {len(temp_batch)} examples")
            temp_batch = []

    # Final flush
    if temp_batch:
        with open(output_json, 'a', encoding='utf-8') as f:
            for ex in temp_batch:
                json.dump(ex, f, ensure_ascii=False)
                f.write("\n")
        print(f"[✓] Final flush: {len(temp_batch)} examples saved")

    # Save Hugging Face dataset
    dataset = Dataset.from_list(all_examples)
    dataset.save_to_disk(output_dir)
    print(f"\n✅ Done: {len(all_examples)} total examples saved to {output_dir}")


if __name__ == "__main__":
    preprocess_articles(
        input_path="ml/data/wikipedia_dataset.json",
        output_dir="ml/data/processed_dataset"
    )