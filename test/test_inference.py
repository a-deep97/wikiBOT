from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

model_path = "ml/checkpoints/checkpoint-82"  
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model.eval()

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    input_ids = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(input_ids, skip_special_tokens=True)
    return answer.strip()

# 🧪 Test Example
context = """
India is a South Asian nation known for its rich cultural heritage, historical depth, and democratic values. With over 1.4 billion people, it is the most populous country in the world. India gained independence from British colonial rule on August 15, 1947, and has since grown into a federal republic with 28 states and 8 union territories, governed from its capital, New Delhi. The country is a melting pot of religions, including Hinduism, Islam, Christianity, Sikhism, Buddhism, and Jainism, reflecting its deep spiritual and philosophical roots. Economically, India is one of the fastest-growing nations, with major sectors in information technology, agriculture, and manufacturing. Its parliamentary democracy is among the largest and most vibrant globally. India is renowned for its landmarks like the Taj Mahal and its colorful festivals such as Diwali and Holi. The nation has made significant advancements in science, space exploration, literature, and cinema. Despite facing challenges such as poverty and inequality, India continues to emerge as a key global player. Its diversity, resilience, and ambition make it a unique and influential presence on the world stage.
"""
question = "What is India?"

print("📌 Question:", question)
print("📚 Context:", context.strip())
print("✅ Answer:", answer_question(question, context))
