import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === Config ===
input_dir = "ml/data/chunked_wiki_articles"
output_dir = "ml/data/wiki_qa_pairs"
os.makedirs(output_dir, exist_ok=True)

model_name = "iarfmoose/t5-base-question-generator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# === Function ===
def generate_qa_pairs(text, max_qas=5):
    prompt = f"generate questions: {text}"
    try:
        outputs = qa_pipeline(prompt, max_length=128, do_sample=False, num_return_sequences=max_qas)
        questions = [out['generated_text'] for out in outputs]
        return [{"question": q.strip(), "context": text} for q in questions]
    except Exception as e:
        print(f"❌ Error during QA generation: {e}")
        return []

# === Main Loop ===
input_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
file_index = 0

for file in input_files:
    with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_path = os.path.join(output_dir, f"qa_{file_index}.jsonl")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(lines, desc=f"Processing {file}"):
            data = json.loads(line)
            context = data["text"]
            title = data["title"]

            qa_pairs = generate_qa_pairs(context)
            for pair in qa_pairs:
                item = {
                    "title": title,
                    "context": pair["context"],
                    "question": pair["question"],
                    "answer": ""  # You may add answer generation later
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_index += 1

print("✅ QA generation completed.")
