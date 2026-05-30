import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch

from src.config import (
    EMBEDDING_MODEL,
    TOP_K,
    FAISS_INDEX_PATH,
    CHUNKS_PATH
)

class HotpotRetriever:

    def __init__(self, chunks=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[INFO] Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

        self.index = None
        self.chunks = chunks or []

    def build_index(self):

        print("[INFO] Building FAISS index...")

        texts = [c["title"] + " " + c["text"] for c in self.chunks]

        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype("float32")

        dim = emb.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb)

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def load(self):

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query, k=TOP_K):

        q_emb = self.model.encode([query]).astype("float32")

        dist, idx = self.index.search(q_emb, k)

        results = []

        for i, d in zip(idx[0], dist[0]):

            c = self.chunks[i]

            results.append({
                "title": c["title"],
                "text": c["text"],
                "score": float(d)
            })

        return results