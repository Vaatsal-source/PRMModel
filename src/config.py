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

TOP_K = 5

# =========================
# RANDOMNESS
# =========================

SEED = 42