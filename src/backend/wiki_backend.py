"""Core backend for wikiBOT."""

import time

from src.data.wikipedia import (
    WikipediaArticleError,
    search_wikipedia_articles,
)

from ..database.sqlite_store import SQLiteStore
from ..knowledge.manager import KnowledgeManager
from ..models.shared_resources import (
    get_shared_embedder,
    get_shared_llm,
)
from .article_context import ArticleContext


class WikiBotBackend:
    """Backend for article loading, retrieval, and question answering."""

    def __init__(
        self,
        model_key: str,
        top_k: int = 3,
        candidate_k: int = 10,
        similarity_threshold: float = 0.35,
    ):
        self.model_key = model_key
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.similarity_threshold = similarity_threshold

        # This connection is owned by this backend instance.
        self.database = SQLiteStore()
        self.knowledge_manager = KnowledgeManager(self.database)

        # Shared model resources.
        self.embedder = get_shared_embedder()
        self.llm = get_shared_llm(model_key)

        # Article-specific state belongs to this backend instance.
        self.article_context = None
        self.current_article_title = None

        # Article autocomplete cache.
        self._suggestion_cache = {}
        self._suggestion_cache_ttl = 300

    @property
    def current_chunk_count(self) -> int:
        """Number of retrieval chunks for the loaded article."""
        if self.article_context is None:
            return 0

        return self.article_context.chunk_count
    
    def load_resources(self):
        """Ensure shared model resources are initialized."""
        if self.embedder is None:
            self.embedder = get_shared_embedder()

        if self.llm is None:
            self.llm = get_shared_llm(self.model_key)

    def suggest_articles(
        self,
        query: str,
        limit: int = 8,
    ) -> list[str]:
        """Return Wikipedia article suggestions with caching."""
        query = query.strip()

        if len(query) < 2:
            return []

        cache_key = (query.lower(), limit)
        current_time = time.monotonic()

        cached_result = self._suggestion_cache.get(cache_key)

        if cached_result is not None:
            cached_time, suggestions = cached_result

            if current_time - cached_time < self._suggestion_cache_ttl:
                return suggestions

        suggestions = search_wikipedia_articles(
            query=query,
            limit=limit,
        )

        self._suggestion_cache[cache_key] = (
            current_time,
            suggestions,
        )

        return suggestions

    def load_knowledge(self, article_title: str):
        """Build an article context before replacing the current one."""
        self.load_resources()

        try:
            chunks = self.knowledge_manager.get_article(article_title)
        except WikipediaArticleError as error:
            raise ValueError(str(error)) from error

        if not chunks:
            raise ValueError(
                f"No chunks found for '{article_title}'."
            )

        new_context = ArticleContext(
            article_title=article_title,
            chunks=chunks,
            embedder=self.embedder,
            llm=self.llm,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            similarity_threshold=self.similarity_threshold,
        )

        # Commit state only after the new context is ready.
        self.article_context = new_context
        self.current_article_title = new_context.article_title

        return chunks

    def get_article_sections(
        self,
        article_title: str | None = None,
    ) -> list[dict[str, str]]:
        """Return sections for the requested or current article."""
        if article_title is None:
            article_title = self.current_article_title

        if not article_title:
            return []

        return self.database.get_article_sections(article_title)

    def ask(self, question: str) -> str:
        """Answer a question using the currently loaded article."""
        if self.article_context is None:
            raise RuntimeError(
                "No knowledge has been loaded. "
                "Call load_knowledge() first."
            )

        return self.article_context.ask(question)

    def ask_with_sources(self, question: str) -> dict:
        """Answer a question and return {"answer", "sources"}."""
        if self.article_context is None:
            raise RuntimeError(
                "No knowledge has been loaded. "
                "Call load_knowledge() first."
            )
        return self.article_context.ask_with_sources(question)

    def close(self):
        """Close backend resources."""
        self.database.close()