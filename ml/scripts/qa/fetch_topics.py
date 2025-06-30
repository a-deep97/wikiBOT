import requests
import os

def fetch_top_wikipedia_articles(year=2025, month=6, day=1, limit=1000):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month:02d}/{day:02d}"
    headers = {
        "User-Agent": "AskWikiBot/1.0 (contact: youremail@example.com)"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

    data = response.json()
    articles = data['items'][0]['articles']

    top_articles = [
        article['article'].replace('_', ' ')
        for article in articles
        if not article['article'].startswith('Special:') and article['article'] != 'Main_Page'
    ]

    return top_articles[:limit]


def save_topics_to_file(titles, output_path="ml/data/topics.txt"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for title in titles:
            f.write(title + '\n')
    print(f"✅ Saved {len(titles)} topics to {output_path}")

if __name__ == "__main__":
    print("📡 Fetching top 500 Wikipedia topics...")
    top_500 = fetch_top_wikipedia_articles()
    save_topics_to_file(top_500)