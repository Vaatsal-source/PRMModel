# PRM-Based Multi-Hop RAG on HotpotQA

A research-oriented implementation of a Process Reward Model (PRM) for multi-hop retrieval-augmented question answering on the HotpotQA distractor setting.

This project investigates whether PRM-guided retrieval pruning can improve retrieval quality, faithfulness, and answer correctness in multi-hop QA pipelines.

---

# Research Objective

This project implements a multi-hop RAG pipeline with a Process Reward Model (PRM) that scores intermediate reasoning steps across retrieval hops.

The system is evaluated on the HotpotQA distractor setting using RAGAS metrics with bootstrap confidence intervals.

Core research hypothesis:

> PRM-guided retrieval pruning improves reasoning trajectory quality and downstream RAG faithfulness in multi-hop QA systems.

---

# Features

* Multi-hop retrieval pipeline
* FAISS-based semantic retrieval
* PRM-guided reasoning-step scoring
* Iterative hop controller
* Bridge entity extraction
* Threshold-based retrieval pruning
* Dual threshold ablations (`t=0.4`, `t=0.6`)
* RAGAS evaluation pipeline
* Bootstrap confidence interval computation
* Failure analysis notebook
* Fully reproducible experimental setup

---

# System Architecture

```text
Question
   ↓
Query Decomposition
   ↓
Hop-k Retrieval
   ↓
PRM Scoring
   ↓
Threshold Pruning
   ↓
Bridge Entity Extraction
   ↓
Next-Hop Retrieval
   ↓
Final Context Aggregation
   ↓
Answer Generation
   ↓
RAGAS Evaluation
```

---

# Tech Stack

## Retrieval

* FAISS
* Sentence Transformers
* BAAI/bge-small-en-v1.5

## PRM

* Cross-Encoder Reranking
* MiniLM / BGE Reranker
* XGBoost Calibration Layer

## QA Generation

* Flan-T5

## Evaluation

* RAGAS

## Dataset

* HotpotQA (Distractor Setting)

---

# Repository Structure

```text
.
├── src/
│   ├── retriever.py
│   ├── prm.py
│   ├── pipeline.py
│   ├── hop_controller.py
│   ├── generator.py
│   ├── utils.py
│   └── config.py
│
├── eval/
│   └── ragas_eval.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── plots/
│   ├── results.json
│   └── results.csv
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── faiss/
│
├── README.md
├── requirements.txt
└── main.py
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repo-url>
cd prm-multihop-rag
```

## 2. Create Python 3.11 Environment

```bash
python3.11 -m venv .venv
```

## 3. Activate Environment

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

## Retrieval Pipeline

```bash
python main.py
```

## RAGAS Evaluation

```bash
python eval/ragas_eval.py
```

---

# Experimental Setup

## Dataset

* HotpotQA Distractor Setting

## Evaluation Size

* 500 held-out questions

## PRM Threshold Ablations

* `t = 0.4`
* `t = 0.6`

## Metrics

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall
* Answer Correctness

## Confidence Intervals

* 95% bootstrap confidence intervals
* Computed over all 500 evaluation samples

---

# Hardware Specifications

## Development Environment

* Intel Iris Xe Graphics
* Python 3.11
* CPU-based local experimentation

## Final Evaluation

* Google Colab GPU runtime

---

# Reproducibility

All experiments use:

* Fixed random seeds
* Deterministic retrieval settings
* Pinned package versions

Random seed:

```text
42
```

---

# Planned Research Analysis

The analysis notebook includes:

* Hop failure analysis
* PRM false-positive pruning examples
* Retrieval drift analysis
* Metric distribution plots
* Threshold ablation comparison

---

# Results

| System        | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
| ------------- | ------------ | ---------------- | ----------------- | -------------- | ------------------ |
| PRM (t = 0.4) | TBD          | TBD              | TBD               | TBD            | TBD                |
| PRM (t = 0.6) | TBD          | TBD              | TBD               | TBD            | TBD                |

---

# License

MIT License
