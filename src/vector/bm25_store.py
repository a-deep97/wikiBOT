import re

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self):
        self.documents = []
        self.bm25 = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    def build(self, documents: list[dict]):
        self.documents = list(documents)

        tokenized_documents = [
            self.tokenize(document["content"])
            for document in self.documents
        ]

        if not tokenized_documents:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[tuple[dict, float]]:

        if self.bm25 is None:
            return []

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )[:top_k]

        results = []

        for index in top_indices:
            results.append(
                (
                    self.documents[index],
                    float(scores[index])
                )
            )

        return results

    def size(self) -> int:
        return len(self.documents)