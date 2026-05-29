from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        print("[INFO] Loading reranker...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        retrieved_chunks,
        top_k=5
    ):

        pairs = [
            (query, chunk["text"])
            for chunk in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        for chunk, score in zip(
            retrieved_chunks,
            scores
        ):
            chunk["rerank_score"] = float(score)

        reranked = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]