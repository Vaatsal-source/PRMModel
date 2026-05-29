from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import os
import pickle
import torch


from src.config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_K,
    FINAL_K,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_PATH
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

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=self.device
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
            "hotpotqa/hotpot_qa",
            "distractor"
        )

        print("[INFO] Dataset loaded.")

    # =====================================
    # SAVE CACHE
    # =====================================

    def save_cache(self, embeddings):

        print("[INFO] Saving retrieval cache...")

        np.save(
            EMBEDDINGS_PATH,
            embeddings
        )

        faiss.write_index(
            self.index,
            str(FAISS_INDEX_PATH)
        )

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        print("[INFO] Cache saved.")

    
    # =====================================
    # LOAD CACHE 
    # =====================================
    def load_cache(self):

        if (
            os.path.exists(EMBEDDINGS_PATH)
            and os.path.exists(FAISS_INDEX_PATH)
            and os.path.exists(CHUNKS_PATH)
        ):

            print("[INFO] Loading cached retrieval artifacts...")

            self.index = faiss.read_index(
               str(FAISS_INDEX_PATH)
            )

            with open(CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)

            print("[INFO] Cache loaded.")

            return True

        return False
    # =====================================
    # BUILD CHUNKS
    # =====================================

    def build_chunks(self, split="validation"):

        print("[INFO] Building paragraph chunks...")

        samples = self.dataset[split]

        chunks = []

        chunk_id = 0

        seen_texts = set()
        for sample in tqdm(samples):

            context = sample["context"]

            titles = context["title"]
            sentences = context["sentences"]

            for title, sent_list in zip(titles, sentences):

                paragraph = " ".join(sent_list)
                
                if paragraph in seen_texts:
                    continue
                seen_texts.add(paragraph)

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

        if self.load_cache():
            return

        print("[INFO] Generating embeddings...")

        texts = [
            f"{chunk['title']} {chunk['text']}"
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        print("[INFO] Creating FAISS index...")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        self.index = index
        self.save_cache(embeddings)
        print("[INFO] FAISS index built.")

    # =====================================
    # RETRIEVE
    # =====================================

    def retrieve(self, query, top_k=RETRIEVAL_K):

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