from sentence_transformers import CrossEncoder

from src.config import (
    RERANKER_MODEL_NAME,
    RERANK_TOP_K
)


class CrossEncoderReranker:

    def __init__(self):

        print(
            "[INFO] Loading reranker..."
        )

        self.model = CrossEncoder(
            RERANKER_MODEL_NAME
        )

    # =====================================
    # RERANK
    # =====================================

    def rerank(
        self,
        question,
        docs,
        top_k=RERANK_TOP_K
    ):

        pairs = []

        for doc in docs:

            pairs.append(
                (
                    question,
                    doc["text"]
                )
            )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for doc, score in zip(
            docs,
            scores
        ):

            item = doc.copy()

            item["rerank_score"] = (
                float(score)
            )

            reranked.append(item)

        reranked.sort(
            key=lambda x:
            x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]