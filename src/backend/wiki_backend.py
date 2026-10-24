from src.database.sqlite_store import SQLiteStore
from src.knowledge.manager import KnowledgeManager
from src.data.wikipedia import WikipediaArticleError
from src.models.shared_resources import (
    get_shared_embedder,
    get_shared_llm,
)
from src.backend.article_context import ArticleContext


class WikiBotBackend:
    """Session-specific wikiBOT backend."""

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

        # Expensive resources are shared through Streamlit's cache.
        self.embedder = get_shared_embedder()
        self.llm = get_shared_llm(model_key)

        # Article-specific state belongs to this backend instance.
        self.article_context = None
        self.current_article_title = None

    def load_resources(self):
        """Resources are initialized by the shared-resource functions."""
        pass

    def load_knowledge(self, article_title: str):
        """Build a complete article context before replacing the old one."""
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

        # Commit state only after all indexes and the pipeline are ready.
        self.article_context = new_context
        self.current_article_title = new_context.article_title

        return chunks

    def ask(self, question: str) -> str:
        if self.article_context is None:
            raise RuntimeError(
                "No knowledge has been loaded. "
                "Call load_knowledge() first."
            )

        return self.article_context.ask(question)

    def get_article_sections(
        self,
        article_title: str | None = None,
    ) -> list[dict[str, str]]:
        title = article_title or self.current_article_title

        if not title:
            return []

        return self.database.get_article_sections(title)

    def close(self):
        self.database.close()
