from pathlib import Path

# Directory Routing
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure all structural directories exist cleanly
FAISS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Reproducibility
SEED = 42  # Hard-pinned global seed

# Model Choices
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QA_MODEL = "deepset/roberta-base-squad2"

# Retrieval Hyperparameters
TOP_K = 10      # Fetch enough to capture distractors + gold items per assignment definition
RERANK_K = 5    # Top candidate pool passed onto the PRM gating mechanisms