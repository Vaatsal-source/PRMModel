from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL_NAME,
    TOP_K
)


class HotpotRetriever:

    def __init__(self):

        print("[INFO] Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.dataset = None
        self.chunks = []
        self.index = None

    # =====================================
    # LOAD DATASET
    # =====================================

    def load_hotpotqa(self):

        print("[INFO] Loading HotpotQA dataset...")

        self.dataset = load_dataset(
            "hotpot_qa",
            "distractor"
        )

        print("[INFO] Dataset loaded.")

    # =====================================
    # BUILD CHUNKS
    # =====================================

    def build_chunks(self, split="validation"):

        print("[INFO] Building paragraph chunks...")

        samples = self.dataset[split]

        chunks = []

        chunk_id = 0

        for sample in tqdm(samples):

            context = sample["context"]

            titles = context["title"]
            sentences = context["sentences"]

            for title, sent_list in zip(titles, sentences):

                paragraph = " ".join(sent_list)

                chunk = {
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": paragraph
                }

                chunks.append(chunk)

                chunk_id += 1

        self.chunks = chunks

        print(f"[INFO] Total chunks: {len(chunks)}")

    # =====================================
    # BUILD FAISS INDEX
    # =====================================

    def build_faiss_index(self):

        print("[INFO] Generating embeddings...")

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        print("[INFO] Creating FAISS index...")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        self.index = index

        print("[INFO] FAISS index built.")

    # =====================================
    # RETRIEVE
    # =====================================

    def retrieve(self, query, top_k=TOP_K):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        retrieved_chunks = []

        for idx, distance in zip(indices[0], distances[0]):

            chunk = self.chunks[idx]

            retrieved_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "distance": float(distance)
            })

        return retrieved_chunks