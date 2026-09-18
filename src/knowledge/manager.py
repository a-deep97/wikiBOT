from datetime import datetime, timezone

from ..database.sqlite_store import SQLiteStore
from ..data.wikipedia import fetch_wikipedia_article
from ..data.processor import chunk_text


class KnowledgeManager:
    """
    Manages Wikipedia knowledge and local caching.
    """

    def __init__(
        self,
        database: SQLiteStore
    ):
        self.database = database

    def get_article(
        self,
        title: str
    ) -> list[dict]:

        # Check SQLite first.
        cached_article = self.database.get_article(
            title
        )

        if cached_article:

            print(
                f"[CACHE] Loading '{title}' from SQLite"
            )

            article_id = cached_article[0]

            return self.database.get_chunks(
                article_id
            )

        # Article doesn't exist locally.
        print(
            f"[WIKIPEDIA] Fetching '{title}'"
        )

        article = fetch_wikipedia_article(
            title
        )

        # Convert article into chunks.
        chunks = chunk_text(
            [article]
        )

        if not chunks:
            raise ValueError(
                "Wikipedia article did not produce "
                "any chunks."
            )

        # Save article.
        article_id = self.database.save_article(
            title=article["title"],
            fetched_at=datetime.now(
                timezone.utc
            ).isoformat()
        )

        # Save chunks.
        self.database.save_chunks(
            article_id,
            chunks
        )

        # Save links.
        self.database.save_links(
            article_id,
            article.get("links", [])
        )

        print(
            f"[CACHE] Saved '{title}' to SQLite"
        )

        return chunks