import wikipediaapi
import os
import time
import re
from collections import deque

def get_wiki_api():
    return wikipediaapi.Wikipedia(
        language='en',
        user_agent='AskWikiBot/1.0 (https://yourdomain.org contact: you@example.com)'
    )

def normalize_title(title):
    return title.strip().lower().replace(' ', '_')

def is_valid_title(title):
    title = title.strip().lower()
    
    # Reject non-article namespaces
    invalid_prefixes = ('file:', 'help:', 'category:', 'template:', 'portal:', 'wikipedia:', 'mediawiki:', 'talk:')
    if title.startswith(invalid_prefixes):
        return False

    # Reject if it starts with dot or is clearly a malformed entry
    if title.startswith('.') or '"' in title or '<' in title or '>' in title:
        return False

    return True  # Only basic filtering

def crawl_wikipedia(seed_topics, max_pages=100, max_links_per_page=50, delay=0.5):
    wiki = get_wiki_api()
    visited = set()
    queue = deque([normalize_title(topic) for topic in seed_topics])
    collected = []

    while queue and len(collected) < max_pages:
        norm_topic = queue.popleft()
        if norm_topic in visited:
            continue

        page = wiki.page(norm_topic)
        if not page.exists():
            continue

        visited.add(norm_topic)

        # Use real page title for output
        page_title = page.title.strip()
        if is_valid_title(page_title.lower()):
            collected.append(page_title)
            print(f"[{len(collected):>5}] Collected: {page_title}")

        # Crawl outgoing links
        links_added = 0
        for linked_title in page.links.keys():
            norm_link = normalize_title(linked_title)
            if norm_link in visited or norm_link in queue:
                continue
            if not is_valid_title(norm_link):
                continue

            queue.append(norm_link)
            links_added += 1
            if links_added >= max_links_per_page:
                break

        time.sleep(delay)

    return collected

def save_titles(titles, output_path="ml/data/wikipedia_topics.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for title in titles:
            f.write(title + '\n')
    print(f"\n✅ Saved {len(titles)} topics to {output_path}")

if __name__ == "__main__":
    seed_domains = [
    "History",
    "Geography",
    "Politics",
    "Science",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Geology",
    "Psychology",
    "Economics",
    "Philosophy",
    "Religion",
    "Sociology",
    "Demographics",
    "Technology",
    "Medicine",
    "Literature",
    "Entertainment",
    "Sports",
    "Astronomy",
    "Engineering",
    "Computer Science",
    "Education",
    "Law",
    "Business",
    "Finance",
    "Linguistics",
    "Art",
    "Architecture",
    "Environmental science",
    "Anthropology",
    "Mythology",
    "Media",
    "Transportation",
    "Military"
]


    print("🌐 Crawling Wikipedia...")
    topics = crawl_wikipedia(seed_topics=seed_domains, max_pages=5000)
    save_titles(topics)
