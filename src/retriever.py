from sentence_transformers import SentenceTransformer
import numpy as np
import torch

from src.config import (
    EMBEDDING_MODEL_NAME,
    TOP_K
)


class HotpotRetriever:

    def __init__(self):

        print("[INFO] Loading embedding model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[INFO] Using device: {self.device}")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=self.device
        )

    # =====================================
    # BUILD DOCS FROM SINGLE HOTPOT SAMPLE
    # =====================================

    def build_docs(self, sample):

        docs = []

        titles = sample["context"]["title"]

        sentences = sample["context"]["sentences"]

        for title, sent_list in zip(
            titles,
            sentences
        ):

            docs.append({
                "title": title,
                "text": " ".join(sent_list)
            })

        return docs

    # =====================================
    # RETRIEVE INSIDE CURRENT SAMPLE ONLY
    # =====================================

    def retrieve(
        self,
        query,
        docs,
        top_k=TOP_K
    ):

        texts = [
            f"{doc['title']} {doc['text']}"
            for doc in docs
        ]

        doc_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = np.dot(
            doc_embeddings,
            query_embedding
        )

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_indices[:top_k]:

            results.append({
                "title": docs[idx]["title"],
                "text": docs[idx]["text"],
                "score": float(scores[idx])
            })

        return results