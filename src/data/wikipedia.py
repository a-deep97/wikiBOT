import re

import wikipediaapi


WIKI_USER_AGENT = "WikipediaRAG/1.0"


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
    Fetch a Wikipedia article and return structured content.

    Returns:

        {
            "title": "...",
            "sections": [
                {
                    "title": "...",
                    "text": "..."
                }
            ]
        }
    """
    wiki = create_wikipedia_client()

    page = wiki.page(title)

    if not page.exists():
        raise ValueError(
            f"Wikipedia page '{title}' not found."
        )

    sections = []

    # The article lead/introduction.
    lead_text = clean_text(page.summary)

    if lead_text:
        sections.append(
            {
                "title": "Introduction",
                "text": lead_text,
            }
        )

    # Main article sections.
    for section in page.sections:
        sections.extend(
            extract_section(section)
        )

    if not sections:
        raise ValueError(
            f"Wikipedia page '{title}' contains no useful content."
        )

    return {
        "title": page.title,
        "sections": sections,
    }
