import wikipediaapi
import json
import os

contact_email = '' # need to add the email address here

def load_titles_from_txt(file_path):
    """Load article titles from a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
    return titles

def fetch_wikipedia_pages(titles, output_path="ml/data/wikipedia_dataset.json"):
    """Fetch full Wikipedia page content and save as JSON."""
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=f'AskWikiBot/1.0 (contact: {contact_email})'
    )
    articles = {}

    for title in titles:
        page = wiki.page(title)
        if page.exists():
            full_text = page.text.strip()
            if full_text:
                articles[title] = full_text
                print(f"[✓] Collected: {title} ({len(full_text)} chars)")
            else:
                print(f"[!] Skipped (empty content): {title}")
        else:
            print(f"[✗] Skipped (not found): {title}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(articles)} articles to {output_path}")

def main():
    topic_file = "ml/data/topics.txt"
    output_file = "ml/data/wikipedia_dataset.json"

    if not os.path.exists(topic_file):
        print(f"[!] Error: {topic_file} not found.")
        return

    titles = load_titles_from_txt(topic_file)
    fetch_wikipedia_pages(titles, output_file)

if __name__ == "__main__":
    main()
