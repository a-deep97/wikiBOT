import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences while keeping the punctuation.
    """
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text
        )
        if sentence.strip()
    ]


def _get_overlap_sentences(
    sentences: list[str],
    overlap_size: int
) -> list[str]:
    """
    Return trailing sentences whose combined length
    is approximately within overlap_size.
    """
    overlap = []
    total_length = 0

    for sentence in reversed(sentences):
        sentence_length = len(sentence)

        if overlap and total_length + sentence_length + 1 > overlap_size:
            break

        overlap.insert(0, sentence)
        total_length += sentence_length + 1

    return overlap


def chunk_section(
    section_title: str,
    section_text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[dict]:

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

    sentences = split_sentences(section_text)

    chunks = []
    current_sentences = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # Handle a single sentence larger than chunk_size.
        if sentence_length > chunk_size:
            if current_sentences:
                chunks.append({
                    "section": section_title,
                    "chunk_index": chunk_index,
                    "content": (
                        f"Section: {section_title}\n"
                        f"{' '.join(current_sentences)}"
                    )
                })

                chunk_index += 1
                current_sentences = []
                current_length = 0

            chunks.append({
                "section": section_title,
                "chunk_index": chunk_index,
                "content": (
                    f"Section: {section_title}\n"
                    f"{sentence}"
                )
            })

            chunk_index += 1
            continue

        new_length = (
            current_length +
            sentence_length +
            (1 if current_sentences else 0)
        )

        if current_sentences and new_length > chunk_size:

            chunks.append({
                "section": section_title,
                "chunk_index": chunk_index,
                "content": (
                    f"Section: {section_title}\n"
                    f"{' '.join(current_sentences)}"
                )
            })

            chunk_index += 1

            overlap_sentences = _get_overlap_sentences(
                current_sentences,
                chunk_overlap
            )

            current_sentences = overlap_sentences
            current_length = sum(
                len(sentence)
                for sentence in current_sentences
            )

            if current_sentences:
                current_length += len(current_sentences) - 1

        current_sentences.append(sentence)

        current_length = sum(
            len(item)
            for item in current_sentences
        )

        if current_sentences:
            current_length += len(current_sentences) - 1

    if current_sentences:
        chunks.append({
            "section": section_title,
            "chunk_index": chunk_index,
            "content": (
                f"Section: {section_title}\n"
                f"{' '.join(current_sentences)}"
            )
        })

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