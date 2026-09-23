import faiss
import numpy as np


class FAISSStore:

    def __init__(self, dimension: int):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[dict]
    ):
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match "
                "number of documents."
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> list[tuple[dict, float]]:

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):
            if index != -1:
                results.append(
                    (
                        self.documents[index],
                        float(score)
                    )
                )

        return results

    def size(self) -> int:
        return self.index.ntotal