import wikipediaapi
import json
import os

contact_email = ''

START_LINE = 0
END_LINE = 5000
CHUNK_SIZE = 500

def load_titles_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def fetch_wikipedia_pages(titles, output_dir, global_offset=0):
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=f'AskWikiBot/1.0 (contact: {contact_email})'
    )

    os.makedirs(output_dir, exist_ok=True)

    for i, title in enumerate(titles):
        global_index = global_offset + i + 1
        chunk_index = (global_index - 1) // CHUNK_SIZE
        chunk_file = os.path.join(output_dir, f"wikipedia_dataset_{chunk_index}.json")

        # Load current chunk state
        articles = {}
        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
            except json.JSONDecodeError:
                print(f"[!] Warning: {chunk_file} is corrupt. Skipping save for now.")
                articles = {}

        if title in articles:
            print(f"[{global_index}][{i}] ↪ Skipped (already saved in chunk {chunk_index}): {title}")
            continue

        page = wiki.page(title)
        if page.exists():
            full_text = page.text.strip()
            if full_text:
                articles[title] = full_text
                print(f"[{global_index}][{i}] ✓ Collected: {title} ({len(full_text)} chars)")

                # Save this topic to its chunk file immediately
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(articles, f, indent=2, ensure_ascii=False)
            else:
                print(f"[{global_index}][{i}] ! Skipped (empty): {title}")
        else:
            print(f"[{global_index}][{i}] ✗ Skipped (not found): {title}")

def main():
    topic_file = "ml/data/wikipedia_topics.txt"
    output_dir = "ml/data/wiki_articles/"

    if not os.path.exists(topic_file):
        print(f"[!] Error: {topic_file} not found.")
        return

    all_titles = load_titles_from_txt(topic_file)
    titles = all_titles[START_LINE:END_LINE]

    print(f"🔄 Processing topics {START_LINE + 1} to {END_LINE} (total: {len(titles)})...\n")
    fetch_wikipedia_pages(titles, output_dir=output_dir, global_offset=START_LINE)

if __name__ == "__main__":
    main()
