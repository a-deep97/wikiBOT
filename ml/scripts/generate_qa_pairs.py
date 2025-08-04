import os
import json
import spacy
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
input_dir = "ml/data/wiki_articles"
output_dir = "ml/data/wiki_qa_pairs"
os.makedirs(output_dir, exist_ok=True)

CHUNK_SIZE = 5000
MAX_QA_PER_ARTICLE = 10
MAX_INPUT_TOKENS = 512

# --- Load spaCy for answer span extraction ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --- Load QA generation model ---
print("CUDA available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else -1
device_str = "cuda" if device == 0 else "cpu"

model_name = "iarfmoose/t5-base-question-generator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device_str)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# --- Helper: extract noun phrases as answers ---
def extract_answers(text, max_answers=5):
    doc = nlp(text)
    return list({chunk.text.strip() for chunk in doc.noun_chunks})[:max_answers]

# --- Generate question-answer pairs ---
def generate_qa_pairs(text, max_qas=10):
    answers = extract_answers(text, max_qas)
    qa_pairs = []

    for ans in answers:
        if ans not in text:
            continue
        highlighted = text.replace(ans, f"<hl> {ans} <hl>", 1)
        prompt = f"generate question: {highlighted}"

        try:
            output = qa_pipeline(prompt, max_length=64, do_sample=False)[0]['generated_text']
            qa_pairs.append({"question": output.strip(), "answer": ans})
        except Exception as e:
            print("Error generating QA:", e)

    return qa_pairs

# --- Incremental saving ---
def save_pair(pair, file_index):
    output_path = os.path.join(output_dir, f"wiki_qa_dataset_{file_index}.jsonl")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

# --- Main loop ---
file_index = 0
pair_count = 0

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
for filename in input_files:
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
        articles = json.load(f)

    for title, content in tqdm(articles.items(), desc=f"Processing {filename}"):
        qa_collected = 0
        text_chunks = [content]  # You can split chunks if needed

        for chunk in text_chunks:
            if qa_collected >= MAX_QA_PER_ARTICLE:
                break

            pairs = generate_qa_pairs(chunk, max_qas=MAX_QA_PER_ARTICLE - qa_collected)
            for pair in pairs:
                data = {
                    "title": title,
                    "context": chunk,
                    "question": pair["question"],
                    "answer": pair["answer"]
                }

                save_pair(data, file_index)
                pair_count += 1
                qa_collected += 1

                if pair_count >= CHUNK_SIZE:
                    file_index += 1
                    pair_count = 0

print("✅ All QA pairs generated and saved incrementally.")
