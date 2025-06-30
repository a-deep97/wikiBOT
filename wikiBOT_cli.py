import click
import wikipediaapi
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from credentials import contact_email

# Load model and tokenizer
MODEL_DIR = "ml/gen_checkpoints/checkpoint-669"  
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()

def fetch_wikipedia_text(title):
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=f'AskWikiBot/1.0 (https://yourdomain.org contact: {contact_email})'
    )
    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"❌ Wikipedia page '{title}' not found.")
    return page.text.strip()

def generate_answer(question, context, max_length=128):
    prompt = f"question: {question} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@click.command()
def askwiki():
    click.secho("🧠 AskWiki (Generative QA CLI)", fg="cyan", bold=True)

    context = None
    title = None

    while True:
        if not context:
            title = click.prompt("📘 Enter Wikipedia Page Title")
            try:
                context = fetch_wikipedia_text(title)
                click.secho(f"✅ Loaded full article for '{title}'", fg="green")
            except Exception as e:
                click.secho(str(e), fg="red")
                continue

        click.secho("\n💬 Ask a question (type 'new' for new topic, or 'exit' to quit):", fg="yellow")
        question = click.prompt("❓ Your question")

        if question.lower() in ["exit", "quit"]:
            click.secho("👋 Goodbye!", fg="cyan")
            break
        elif question.lower() == "new":
            context = None
            continue

        try:
            answer = generate_answer(question, context)
            click.secho(f"🧠 Answer: {answer}\n", fg="green")
        except Exception as e:
            click.secho(f"⚠️ Error: {str(e)}", fg="red")

if __name__ == "__main__":
    askwiki()
