from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_K

class Reranker:

    def __init__(self):

        print("[INFO] Loading reranker...")
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query, docs):

        pairs = []

        for d in docs:
            text = d["title"] + " " + d["text"]
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        docs = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return docs[:RERANK_K]