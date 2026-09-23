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
        llm: Model,
        top_k: int = 3,
        similarity_threshold: float = 0.35
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, question: str) -> list[dict]:

        question_embedding = self.embedder.embed_text(
            question
        )

        results = self.vector_store.search(
            question_embedding,
            top_k=self.top_k
        )

        filtered_results = [
            document
            for document, score in results
            if score >= self.similarity_threshold
        ]

        return filtered_results

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
