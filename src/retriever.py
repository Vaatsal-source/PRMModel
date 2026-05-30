import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, FAISS_DIR

class HotpotRetriever:
    def __init__(self, chunks=None):
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = chunks if chunks is not None else []
        self.index = None
        self.index_path = FAISS_DIR / "index.faiss"
        self.chunks_path = FAISS_DIR / "chunks.pkl"

    def build_index(self):
        if not self.chunks:
            raise ValueError("Cannot build index with empty text chunks source.")
        
        print("[INFO] Encoding document corpus into FAISS matrix...")
        # Extract plain strings for encoding
        texts = [f"{c['title']} {c['text']}" for c in self.chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product / Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save to disk for state consistency
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print("[INFO] FAISS Index successfully written to disk.")

    def load_index(self):
        if self.index_path.exists() and self.chunks_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            print("[INFO] FAISS Index and chunk metadata loaded from cache.")
            return True
        return False

    def retrieve(self, query: str, top_k: int = 10) -> list:
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("FAISS Index is uninitialized. Run build_index() first.")
                
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            chunk_meta = self.chunks[idx].copy()
            chunk_meta["retrieval_score"] = float(score)
            results.append(chunk_meta)
        return results