import time

from src.data.wikipedia import (
    WikipediaArticleError,
    search_wikipedia_articles,
)

from ..database.sqlite_store import SQLiteStore
from ..knowledge.manager import KnowledgeManager
from ..embeddings.embedder import Embedder
from ..vector.faiss_store import FAISSStore
from ..vector.bm25_store import BM25Store
from ..models.model import Model
from ..pipeline import RAGPipeline


class WikiBotBackend:
    """
    Core backend for wikiBOT.

    Owns the knowledge store, embedding model,
    LLM, retrieval stores, and RAG pipeline.
    """

    def __init__(
        self,
        model_key: str,
        top_k: int = 3,
        candidate_k: int = 10,
        similarity_threshold: float = 0.35
    ):
        self.model_key = model_key

        self.top_k = top_k
        self.candidate_k = candidate_k
        self.similarity_threshold = similarity_threshold

        # Persistent knowledge store
        self.database = SQLiteStore()

        self.knowledge_manager = KnowledgeManager(
            self.database
        )

        # Currently loaded article
        self.current_article_title = None

        # Shared ML resources
        self.embedder = None
        self.llm = None

        # Article-specific retrieval resources
        self.vector_store = None
        self.bm25_store = None

        # Final RAG pipeline
        self.pipeline = None

        # Article autocomplete cache
        self._suggestion_cache = {}
        self._suggestion_cache_ttl = 300

    # --------------------------------------------------
    # Shared resources
    # --------------------------------------------------

    def load_resources(self):
        """Load the embedding model and LLM."""

        if self.embedder is None:
            print("[BACKEND] Loading embedder...")

            start = time.perf_counter()

            self.embedder = Embedder()

            print(
                f"[TIMING] Embedder initialization: "
                f"{time.perf_counter() - start:.2f}s"
            )

        if self.llm is None:
            print(
                f"[BACKEND] Loading LLM: "
                f"{self.model_key}"
            )

            start = time.perf_counter()

            self.llm = Model(self.model_key)

            print(
                f"[TIMING] LLM initialization: "
                f"{time.perf_counter() - start:.2f}s"
            )

    # --------------------------------------------------
    # Knowledge loading
    # --------------------------------------------------

    def load_knowledge(
        self,
        article_title: str
    ):
        """
        Load an article and build its retrieval indexes.
        """

        total_start = time.perf_counter()

        # Make sure shared models are available
        self.load_resources()

        # ----------------------------------------------
        # 1. Load article from SQLite/Wikipedia
        # ----------------------------------------------

        start = time.perf_counter()

        try:
            chunks = self.knowledge_manager.get_article(
                article_title
            )

        except WikipediaArticleError as error:

            raise ValueError(
                str(error)
            ) from error

        if not chunks:
            raise ValueError(
                f"No chunks found for '{article_title}'."
            )

        # Store the successfully loaded article
        self.current_article_title = article_title.strip()

        print(
            f"[TIMING] SQLite/Wikipedia: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # ----------------------------------------------
        # 2. Generate embeddings
        # ----------------------------------------------

        start = time.perf_counter()

        chunk_texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_documents(
            chunk_texts
        )

        print(
            f"[TIMING] Embedding generation "
            f"({len(chunks)} chunks): "
            f"{time.perf_counter() - start:.2f}s"
        )

        # ----------------------------------------------
        # 3. Build FAISS
        # ----------------------------------------------

        start = time.perf_counter()

        self.vector_store = FAISSStore(
            dimension=embeddings.shape[1]
        )

        self.vector_store.add(
            embeddings,
            chunks
        )

        print(
            f"[TIMING] FAISS: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # ----------------------------------------------
        # 4. Build BM25
        # ----------------------------------------------

        start = time.perf_counter()

        self.bm25_store = BM25Store()
        self.bm25_store.build(chunks)

        print(
            f"[TIMING] BM25: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # ----------------------------------------------
        # 5. Build RAG pipeline
        # ----------------------------------------------

        self.pipeline = RAGPipeline(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            llm=self.llm,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            similarity_threshold=(
                self.similarity_threshold
            )
        )

        print(
            f"[TIMING] TOTAL knowledge load: "
            f"{time.perf_counter() - total_start:.2f}s"
        )

        return chunks

    # --------------------------------------------------
    # Article sections
    # --------------------------------------------------

    def get_article_sections(
        self,
        article_title: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Return cached sections for an article.

        If no article title is supplied, return sections
        for the currently loaded article.
        """

        if article_title is None:
            article_title = self.current_article_title

        if not article_title:
            return []

        return self.database.get_article_sections(
            article_title
        )

    # --------------------------------------------------
    # Question answering
    # --------------------------------------------------

    def ask(self, question: str) -> str:
        """Ask a question using the loaded knowledge."""

        if self.pipeline is None:
            raise RuntimeError(
                "No knowledge has been loaded. "
                "Call load_knowledge() first."
            )

        return self.pipeline.ask(question)

    # --------------------------------------------------
    # Article autocomplete
    # --------------------------------------------------

    def suggest_articles(
        self,
        query: str,
        limit: int = 8
    ) -> list[str]:
        """
        Return Wikipedia article suggestions for a query.

        Results are cached for a short period to avoid
        repeatedly querying Wikipedia.
        """

        query = query.strip()

        if len(query) < 2:
            return []

        cache_key = (query.lower(), limit)
        current_time = time.monotonic()

        cached_result = self._suggestion_cache.get(
            cache_key
        )

        if cached_result:
            cached_time, suggestions = cached_result

            if (
                current_time - cached_time
                < self._suggestion_cache_ttl
            ):
                return suggestions

        suggestions = search_wikipedia_articles(
            query=query,
            limit=limit
        )

        self._suggestion_cache[cache_key] = (
            current_time,
            suggestions
        )

        return suggestions

    # --------------------------------------------------
    # Resource cleanup
    # --------------------------------------------------

    def close(self):
        """Close backend resources."""

        self.database.close()