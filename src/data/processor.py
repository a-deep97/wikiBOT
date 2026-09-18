import re


def clean_text(text: str) -> str:
    """
    Clean Wikipedia text.
    """
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_section(
    section_title: str,
    section_text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Split one section into chunks.

    Returns structured chunks containing:
        - section
        - chunk_index
        - content
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
    chunk_index = 0

    step = chunk_size - chunk_overlap

    while start < len(section_text):

        end = start + chunk_size

        text = section_text[start:end].strip()

        if text:
            chunks.append({
                "section": section_title,
                "chunk_index": chunk_index,
                "content": (
                    f"Section: {section_title}\n"
                    f"{text}"
                )
            })

            chunk_index += 1

        start += step

    return chunks


def chunk_text(
    articles: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[dict]:

    chunks = []

    for article in articles:

        article_title = article.get("title", "")

        sections = article.get("sections", [])

        for section in sections:

            section_title = section.get("title", "")
            section_text = section.get("text", "")

            if not section_text:
                continue

            section_chunks = chunk_section(
                section_title=section_title,
                section_text=section_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            for chunk in section_chunks:

                chunk["content"] = (
                    f"Article: {article_title}\n"
                    f"{chunk['content']}"
                )

                chunks.append(chunk)

    return chunks