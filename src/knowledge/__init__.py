from datetime import datetime, timezone

from ..data.wikipedia import fetch_wikipedia_article
from ..data.processor import chunk_text
from ..database.sqlite_store import SQLiteStore


class KnowledgeManager:
    """
    Manages Wikipedia content and the local cache.
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
        """
        Get article chunks.

        Use local cache when available.
        Otherwise fetch from Wikipedia.
        """

        cached_article = self.database.get_article(
            title
        )

        if cached_article:

            article_id = cached_article[0]

            return self.database.get_chunks(
                article_id
            )

        # Article not cached.
        articles = fetch_wikipedia_article(
            title,
            max_linked_pages=0
        )

        if not articles:
            return []

        article = articles[0]

        chunks = chunk_text(
            [article]
        )

        article_id = self.database.save_article(
            title=article["title"],
            fetched_at=datetime.now(
                timezone.utc
            ).isoformat()
        )

        self.database.save_chunks(
            article_id,
            chunks
        )

        self.database.save_links(
            article_id,
            article.get("links", [])
        )

        return chunks