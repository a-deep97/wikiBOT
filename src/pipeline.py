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

        # -----------------------------
        # FAISS semantic search
        # -----------------------------

        question_embedding = self.embedder.embed_text(
            question
        )

        faiss_results = self.vector_store.search(
            question_embedding,
            top_k=self.candidate_k
        )

        faiss_results = [
            (document, score)
            for document, score in faiss_results
            if score >= self.similarity_threshold
        ]

        # -----------------------------
        # BM25 keyword search
        # -----------------------------

        bm25_results = self.bm25_store.search(
            question,
            top_k=self.candidate_k
        )

        # -----------------------------
        # Merge results
        # -----------------------------

        candidates = {}

        for document, _ in faiss_results:
            key = self._document_key(document)
            candidates[key] = document

        for document, _ in bm25_results:
            key = self._document_key(document)
            candidates[key] = document

        # -----------------------------
        # Return final context
        # -----------------------------

        return list(candidates.values())[:self.top_k]

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
