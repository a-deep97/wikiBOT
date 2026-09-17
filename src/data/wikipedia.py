import wikipediaapi

WIKI_USER_AGENT = "WikipediaRAG/1.0"

def fetch_wikipedia_text(title: str) -> str:
    """
    Fetch the text of a Wikipedia article.

    ```
    Args:
        title: Wikipedia article title.

    Returns:
        Article text.

    Raises:
        ValueError: If the Wikipedia page does not exist or is empty.
    """

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent=WIKI_USER_AGENT
    )

    page = wiki.page(title)

    if not page.exists():
        raise ValueError(
            f"Wikipedia page '{title}' not found."
        )

    text = page.text.strip()

    if not text:
        raise ValueError(
            f"Wikipedia page '{title}' is empty."
        )

    return text

