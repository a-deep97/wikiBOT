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
        top_k: int = 3
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def retrieve(self, question: str) -> list[str]:
        """
        Retrieve relevant document chunks for a question.

        Args:
            question: User's question.

        Returns:
            Relevant document chunks.
        """

        question_embedding = self.embedder.embed_text(
            question
        )

        documents = self.vector_store.search(
            question_embedding,
            top_k=self.top_k
        )

        return documents

    def build_prompt(
        self,
        question: str,
        context: list[str]
        ) -> str:
        """
        Build the prompt sent to the LLM.
        """

        context_text = "\n\n".join(context)

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
        """
        Retrieve relevant context and generate an answer.

        Args:
            question: User's question.

        Returns:
            Generated answer.
        """

        context = self.retrieve(question)

        if not context:
            return "I could not find relevant information."

        prompt = self.build_prompt(
            question,
            context
        )

        return self.llm.generate(prompt)
