import re

import requests
import wikipediaapi


WIKI_USER_AGENT = "AskWikiBot/1.0"


UNWANTED_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
}


class WikipediaArticleError(Exception):
    """Raised when a Wikipedia article cannot be loaded."""


def create_wikipedia_client() -> wikipediaapi.Wikipedia:
    """
    Create a Wikipedia API client.
    """
    return wikipediaapi.Wikipedia(
        language="en",
        user_agent=WIKI_USER_AGENT
    )


def clean_text(text: str) -> str:
    """
    Clean Wikipedia text while preserving readable sentences.
    """
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_section_title(title: str) -> str:
    """
    Normalize a section title.
    """
    return title.strip()


def is_unwanted_section(title: str) -> bool:
    """
    Check whether a section should be excluded.
    """
    normalized_title = title.strip().lower()

    return normalized_title in UNWANTED_SECTIONS


def extract_section(
    section,
    parent_title: str = ""
) -> list[dict[str, str]]:
    """
    Recursively extract useful Wikipedia sections.

    Subsections are also included.
    """
    sections = []

    title = clean_section_title(section.title)

    # Ignore unwanted sections completely.
    if is_unwanted_section(title):
        return sections

    text = clean_text(section.text)

    if text:
        if parent_title:
            full_title = f"{parent_title} > {title}"
        else:
            full_title = title

        sections.append(
            {
                "title": full_title,
                "text": text,
            }
        )

    for subsection in section.sections:
        sections.extend(
            extract_section(
                subsection,
                parent_title=title
            )
        )

    return sections


def fetch_wikipedia_article(title: str) -> dict:
    """
    Fetch and validate a Wikipedia article.

    Raises:
        WikipediaArticleError:
            If the title is empty, the article does not exist,
            or the title refers to a disambiguation page.
    """

    title = title.strip()

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if not title:
        raise WikipediaArticleError(
            "Wikipedia article title cannot be empty."
        )

    # --------------------------------------------------
    # Create Wikipedia client
    # --------------------------------------------------

    wiki = create_wikipedia_client()

    # --------------------------------------------------
    # Fetch article
    # --------------------------------------------------

    page = wiki.page(title)

    # --------------------------------------------------
    # Article does not exist
    # --------------------------------------------------

    if not page.exists():
        raise WikipediaArticleError(
            f"Wikipedia article '{title}' does not exist."
        )

    # --------------------------------------------------
    # Extract useful sections
    # --------------------------------------------------

    sections = []

    # Wikipedia's lead/introduction is stored separately from page.sections.
    if page.summary.strip():
        sections.append(
            {
                "title": "Introduction",
                "text": clean_text(page.summary),
            }
        )

    for section in page.sections:
        sections.extend(extract_section(section))

    # --------------------------------------------------
    # Extract links
    # --------------------------------------------------

    links = [
        link.title
        for link in page.links.values()
    ]

    # --------------------------------------------------
    # Return article
    # --------------------------------------------------

    return {
        "title": page.title,
        "summary": page.summary,
        "sections": sections,
        "links": links,
    }

def search_wikipedia_articles(
    query: str,
    limit: int = 8
) -> list[str]:
    """
    Search Wikipedia for article title suggestions.

    Returns a list of matching article titles.
    """

    query = query.strip()

    if len(query) < 2:
        return []

    if limit < 1:
        return []

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={
                "User-Agent": WIKI_USER_AGENT
            },
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not search Wikipedia: {error}"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "Wikipedia returned an invalid response."
        ) from error

    results = data.get("query", {}).get("search", [])

    return [
        result["title"]
        for result in results
        if result.get("title")
    ]