import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

from src.config import DATA_DIR, RESULTS_DIR, SEED
from src.pipeline import MultiHopQAPipeline

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_bootstrap_ci(scores, n_bootstrap=1000, alpha=0.95):
    """Computes a 95% bootstrap confidence interval over an array of scores."""
    rng = np.random.default_rng(SEED)
    means = []
    n = len(scores)
    
    # Bootstrap resampling with replacement
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))
        
    low_percentile = ((1.0 - alpha) / 2.0) * 100
    high_percentile = (alpha + (1.0 - alpha) / 2.0) * 100
    
    low = np.percentile(means, low_percentile)
    high = np.percentile(means, high_percentile)
    return low, high

def main():
    seed_everything(SEED)
    
    heldout_path = DATA_DIR / "heldout_500.json"
    with open(heldout_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Threshold configurations to test
    prm_thresholds = [0.4, 0.6]
    summary_results = []

    for t in prm_thresholds:
        print(f"\n{'='*50}\n[EVAL] Evaluating PRM Configuration: t = {t}\n{'='*50}")
        
        # Initialize pipeline (will auto-load global index from disk)
        pipeline = MultiHopQAPipeline(chunks=None, threshold=t)
        
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        # Run inference across all evaluation questions
        for sample in tqdm(samples, desc=f"Running Pipeline (t={t})"):
            q = sample["question"]
            gold_ans = sample["answer"]
            
            # Execute pipeline
            output = pipeline.run_pipeline(q)
            
            # Gather extracted contexts for RAGAS evaluation
            retrieved_contexts = [doc["text"] for doc in output["evidence"]]
            
            ragas_data["question"].append(q)
            ragas_data["answer"].append(output["predicted_answer"])
            ragas_data["contexts"].append(retrieved_contexts)
            ragas_data["ground_truth"].append(gold_ans)

        # Convert to Hugging Face Dataset for RAGAS evaluation ingestion
        hf_dataset = Dataset.from_dict(ragas_data)
        
        print(f"[INFO] Computing individual RAGAS metric scores for t={t}...")
        # Evaluate using all 5 mandatory metrics
        ragas_output = evaluate(
            dataset=hf_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness
            ]
        )
        
        # Extract row-level pandas dataframe for bootstrapping intervals
        df_scores = ragas_output.to_pandas()
        
        # Save raw outputs for verification requirements
        raw_export_path = RESULTS_DIR / f"raw_scores_t_{t}.csv"
        df_scores.to_csv(raw_export_path, index=False)
        print(f"[INFO] Raw scores saved directly to {raw_export_path}")

        metrics_list = [
            "faithfulness", 
            "answer_relevancy", 
            "context_precision", 
            "context_recall", 
            "answer_correctness"
        ]
        
        t_summary = {"System": f"Your system (PRM t={t})"}
        
        # Run bootstrap statistical analysis across all metrics
        for metric in metrics_list:
            # Drop any NaNs defensively to avoid spreading errors
            scores_array = df_scores[metric].dropna().to_numpy()
            
            mean_val = np.mean(scores_array)
            low_ci, high_ci = compute_bootstrap_ci(scores_array)
            
            # Format target string: "Mean [95% CI_low - CI_high]"
            t_summary[metric] = f"{mean_val:.3f} [{low_ci:.3f} - {high_ci:.3f}]"
            
        summary_results.append(t_summary)

    # Consolidated formatting of the target results table
    df_final_table = pd.DataFrame(summary_results)
    
    # Match assignment column header name naming specs
    df_final_table.columns = ["System", "Faith.", "Ans. Rel.", "Ctx. Prec.", "Ctx. Rec.", "Ans. Corr."]
    
    # Save target report assets
    df_final_table.to_csv(RESULTS_DIR / "results.csv", index=False)
    df_final_table.to_json(RESULTS_DIR / "results.json", orient="records", indent=2)
    
    print("\n" + "#"*60 + "\nFINAL EVALUATION RESULTS TABLE\n" + "#"*60)
    print(df_final_table.to_string(index=False))
    print("#"*60)

if __name__ == "__main__":
    main()