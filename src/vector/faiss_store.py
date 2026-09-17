import faiss
import numpy as np


class FAISSStore:
    """
    Stores document embeddings and performs similarity search.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Keep the original documents corresponding to vectors
        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[str]
    ):
        """
        Add document embeddings to the FAISS index.
        """

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 2
    ) -> list[str]:
        """
        Search for the most similar documents.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_embedding = query_embedding.reshape(1, -1)

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)

        _, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for index in indices[0]:
            if index != -1:
                results.append(self.documents[index])

        return results

    def size(self) -> int:
        return self.index.ntotal