import os
import json
from pathlib import Path
import spacy

import torch

import pdb
pdb.set_trace()
if torch.cuda.is_available():
    print("✅ GPU is available.")
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not available, using CPU.")

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Paths
INPUT_DIR = "ml/data/wiki_articles"
OUTPUT_DIR = "ml/data/chunked_wiki_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
CHUNK_WORD_LIMIT = 150

def sent_tokenize(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def split_into_chunks(text, max_words=150):
    """Split text into sentence-based chunks of approximately max_words length."""
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_words = [], [], 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_words + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_words = word_count
        else:
            current_chunk.append(sent)
            current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_file(input_path, output_path):
    """Read one JSON file of articles, chunk them, and write to JSONL."""
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    with open(output_path, "w", encoding="utf-8") as out_file:
        for title, text in articles.items():
            chunks = split_into_chunks(text, max_words=CHUNK_WORD_LIMIT)
            for idx, chunk in enumerate(chunks):
                item = {
                    "title": title,
                    "chunk_id": idx,
                    "text": chunk
                }
                out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    for file_name in input_files:
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, f"chunked_{Path(file_name).stem}.jsonl")
        print(f"🔄 Processing {file_name} → {os.path.basename(output_path)}")
        process_file(input_path, output_path)

    print("✅ All articles chunked and saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
