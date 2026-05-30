from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss"

FAISS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================
# MODELS
# =====================================

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# =====================================
# RETRIEVAL
# =====================================

RETRIEVAL_K = 10

RERANK_TOP_K = 3

MAX_HOPS = 3

# =====================================
# RANDOM
# =====================================

SEED = 42