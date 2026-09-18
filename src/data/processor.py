import re


def clean_text(text: str) -> str:
    """
    Clean Wikipedia text.
    """
    # Replace multiple whitespace characters with one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_section(
    section_title: str,
    section_text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Split one Wikipedia section into chunks.

    The section title is included in every chunk so that
    the embedding contains information about the context.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    section_text = clean_text(section_text)

    if not section_text:
        return []

    chunks = []

    start = 0
    step = chunk_size - chunk_overlap

    while start < len(section_text):

        end = start + chunk_size

        text = section_text[start:end].strip()

        if text:
            chunk = (
                f"Section: {section_title}\n"
                f"{text}"
            )

            chunks.append(chunk)

        start += step

    return chunks


def chunk_text(
    article: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Convert a structured Wikipedia article into chunks.

    Expected article format:

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
    if not article:
        return []

    sections = article.get("sections", [])

    if not sections:
        return []

    chunks = []

    for section in sections:

        section_title = section.get("title", "").strip()
        section_text = section.get("text", "").strip()

        if not section_text:
            continue

        section_chunks = chunk_section(
            section_title=section_title,
            section_text=section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks.extend(section_chunks)

    return chunks
