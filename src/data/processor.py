import re


def clean_text(text: str) -> str:
    """
    Clean raw Wikipedia article text.

    Args:
        text: Raw article text.

    Returns:
        Cleaned text.
    """

    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        List of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks

