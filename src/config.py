from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5
RERANK_K = 3

FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
CHUNKS_PATH = FAISS_DIR / "chunks.pkl"

FAISS_DIR.mkdir(parents=True, exist_ok=True)