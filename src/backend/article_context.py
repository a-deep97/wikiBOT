
from src.pipeline import RAGPipeline
from src.vector.faiss_store import FAISSStore
from src.vector.bm25_store import BM25Store


class ArticleContext:
    """Own retrieval resources for one loaded article."""

    def __init__(
        self,
        article_title: str,
        chunks: list[dict],
        embedder,
        llm,
        top_k: int = 3,
        candidate_k: int = 10,
        similarity_threshold: float = 0.35,
    ):
        self.article_title = article_title.strip()

        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = embedder.embed_documents(chunk_texts)

        vector_store = FAISSStore(
            dimension=embeddings.shape[1]
        )
        vector_store.add(embeddings, chunks)

        bm25_store = BM25Store()
        bm25_store.build(chunks)

        self.pipeline = RAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            bm25_store=bm25_store,
            llm=llm,
            top_k=top_k,
            candidate_k=candidate_k,
            similarity_threshold=similarity_threshold,
        )

    def ask(self, question: str) -> str:
        return self.pipeline.ask(question)
