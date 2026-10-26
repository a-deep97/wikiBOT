from src.vector.bm25_store import BM25Store

from .embeddings.embedder import Embedder
from .vector.faiss_store import FAISSStore
from .models.model import Model


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Flow:

        Question
            ↓
        Embedding
            ↓
        FAISS retrieval
            ↓
        Context
            ↓
        Prompt
            ↓
        LLM
            ↓
        Answer
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: FAISSStore,
        bm25_store: BM25Store,
        llm: Model,
        top_k: int = 3,
        candidate_k: int = 10,
        similarity_threshold: float = 0.35
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.llm = llm
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _document_key(document: dict) -> tuple:
        return (
            document.get("section", ""),
            document.get("chunk_index", -1),
            document.get("content", "")
        )

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve and rank chunks using FAISS and BM25 with RRF."""

        # 1. FAISS semantic search
        question_embedding = self.embedder.embed_text(question)

        faiss_results = self.vector_store.search(
            question_embedding,
            top_k=self.candidate_k,
        )

        # Preserve the existing similarity threshold.
        faiss_results = [
            document
            for document, score in faiss_results
            if score >= self.similarity_threshold
        ]

        # 2. BM25 keyword search
        bm25_results = self.bm25_store.search(
            question,
            top_k=self.candidate_k,
        )

        # 3. Merge candidates and accumulate reciprocal rank scores.
        # Ranks are 1-based. Each chunk is scored once per retrieval list.
        candidates = {}
        scores = {}

        def add_results(results: list[dict]) -> None:
            for rank, document in enumerate(results, start=1):
                key = self._document_key(document)

                if key not in candidates:
                    candidates[key] = document
                    scores[key] = 0.0

                scores[key] += 1.0 / (60 + rank)

        add_results(faiss_results)
        add_results([document for document, _ in bm25_results])

        # 4. Return the highest-ranked unique chunks.
        ranked_keys = sorted(
            candidates,
            key=lambda key: scores[key],
            reverse=True,
        )

        return [
            candidates[key]
            for key in ranked_keys[:self.top_k]
        ]

    def build_prompt(
        self,
        question: str,
        context: list[dict]
    ) -> str:

        context_text = "\n\n".join(
            chunk["content"]
            for chunk in context
        )

        prompt = f"""
            Question: {question}

            Use the following context to answer the question.
            If the answer is not present in the context, say that you do not know.

            Context:
            {context_text}

            Answer:
        """

        return prompt.strip()

    def ask(self, question: str) -> str:

        context = self.retrieve(question)

        print("\n--- Retrieved Context ---")

        for i, chunk in enumerate(context):
            print(f"\n[Chunk {i + 1}]")
            print(chunk["content"][:500])

        print("\n-------------------------")

        if not context:
            return "I could not find relevant information."

        prompt = self.build_prompt(
            question,
            context
        )

        print("\n--- Prompt ---")
        print(prompt)
        print("\n--------------")

        return self.llm.generate(prompt)

    def ask_with_sources(self, question: str) -> dict:
        context = self.retrieve(question)
        if not context:
            return {"answer": "I could not find relevant information.", "sources": []}
        prompt = self.build_prompt(question, context)
        return {"answer": self.llm.generate(prompt), "sources": context}
