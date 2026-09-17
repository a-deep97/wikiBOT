from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Converts text into numerical embedding vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        """
        Convert a single piece of text into an embedding.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """

        return self.model.encode(
            text,
            convert_to_numpy=True
        )

    def embed_documents(self, texts: list[str]):
        """
        Convert multiple documents/chunks into embeddings.

        Args:
            texts: List of text chunks.

        Returns:
            NumPy array containing embeddings.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True
        )
