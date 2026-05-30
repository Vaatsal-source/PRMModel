from sentence_transformers import SentenceTransformer
import torch

from src.config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_K
)


class HotpotRetriever:

    def __init__(self):

        print("[INFO] Loading embedding model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[INFO] Using device: {self.device}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=self.device
        )

    # =====================================
    # BUILD DOCS FROM HOTPOT SAMPLE
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
    # RETRIEVE
    # =====================================

    def retrieve(
        self,
        question,
        docs,
        top_k=RETRIEVAL_K
    ):

        corpus = [

            f"{doc['title']} {doc['text']}"

            for doc in docs
        ]

        doc_embeddings = self.model.encode(
            corpus,
            convert_to_tensor=True
        )

        query_embedding = self.model.encode(
            question,
            convert_to_tensor=True
        )

        scores = (
            query_embedding
            @ doc_embeddings.T
        )

        scores = scores.cpu().tolist()

        retrieved = []

        for doc, score in zip(
            docs,
            scores
        ):

            retrieved.append({
                "title": doc["title"],
                "text": doc["text"],
                "score": float(score)
            })

        retrieved.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return retrieved[:top_k]