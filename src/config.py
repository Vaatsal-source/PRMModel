from pathlib import Path

# =====================================
# PATHS
# =====================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =====================================
# MODEL CONFIG
# =====================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# =====================================
# RETRIEVAL
# =====================================

TOP_K = 3

# =====================================
# RANDOMNESS
# =====================================

SEED = 42