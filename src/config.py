from pathlib import Path

# =========================
# PATHS
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FAISS_DIR = DATA_DIR / "faiss"

RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

# =========================
# MODELS
# =========================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# =========================
# RETRIEVAL
# =========================

RETRIEVAL_K = 20
FINAL_K = 5

# =========================
# RANDOMNESS
# =========================

SEED = 42

EMBEDDINGS_PATH = FAISS_DIR / "embeddings.npy"
FAISS_INDEX_PATH = FAISS_DIR / "hotpot.index"
CHUNKS_PATH = FAISS_DIR / "chunks.pkl"

FAISS_DIR.mkdir(parents=True, exist_ok=True)