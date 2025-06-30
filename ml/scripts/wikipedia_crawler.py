import wikipediaapi
import os
import time
from collections import deque

def get_wiki_api():
    return wikipediaapi.Wikipedia(
        language='en',
        user_agent='AskWikiBot/1.0 (https://yourdomain.org contact: you@example.com)'
    )

def crawl_wikipedia(seed_topics, max_pages=10000, max_links_per_page=50, delay=0.5):
    wiki = get_wiki_api()
    visited = set()
    queue = deque(seed_topics)
    collected = []

    while queue and len(collected) < max_pages:
        topic = queue.popleft()

        if topic in visited:
            continue

        page = wiki.page(topic)
        if not page.exists():
            continue

        visited.add(topic)
        collected.append(topic)
        print(f"[{len(collected):>5}] Collected: {topic}")

        # Crawl links from this page
        linked_titles = list(page.links.keys())[:max_links_per_page]
        for link in linked_titles:
            if link not in visited and link not in queue:
                queue.append(link)

        time.sleep(delay)  # Be respectful to Wikipedia API

    return collected

def save_titles(titles, output_path="ml/data/wikipedia_topics.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for title in titles:
            f.write(title + '\n')
    print(f"\n✅ Saved {len(titles)} topics to {output_path}")

if __name__ == "__main__":
    # Seed topics — start here
    seed = [
        "India", "Computer", "Science", "Philosophy", "Technology", "Mathematics",
        "Artificial intelligence", "Physics", "Chemistry", "History",
        "Economics", "Biology", "Psychology", "Engineering", "Literature"
    ]

    print("🌐 Crawling Wikipedia...")
    topics = crawl_wikipedia(seed_topics=seed, max_pages=7000)
    save_titles(topics)
