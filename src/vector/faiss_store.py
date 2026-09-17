import faiss
import numpy as np


class FAISSStore:
    """
    Stores document embeddings and performs similarity search.
    """

    def __init__(self, dimension: int):
        """
        Create a FAISS vector store.

        Args:
            dimension: Dimension of the embedding vectors.
        """

        self.dimension = dimension

        # L2 distance index.
        self.index = faiss.IndexFlatL2(dimension)

        # Keep the original text corresponding to each vector.
        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[str]
    ):
        """
        Add document embeddings to the FAISS index.

        Args:
            embeddings: Embedding vectors.
            documents: Original text corresponding to each vector.
        """

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> list[str]:
        """
        Find the most relevant documents.

        Args:
            query_embedding: Embedding of the user's question.
            top_k: Number of documents to retrieve.

        Returns:
            List of relevant documents.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for index in indices[0]:
            if index != -1:
                results.append(self.documents[index])

        return results

    def size(self) -> int:
        """
        Return the number of stored vectors.
        """

        return self.index.ntotal