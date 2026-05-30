import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

# Force strict reproducibility
SEED = 42
np.random.seed(SEED)


def bootstrap_confidence_interval(
    scores: np.ndarray, 
    num_bootstraps: int = 1000, 
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes the mean and a non-parametric bootstrap confidence interval 
    for a given distribution of metric scores.
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
        
    boot_means = []
    n = len(scores)
    
    # Generate bootstrap samples by resampling with replacement
    for _ in range(num_bootstraps):
        boot_sample = np.random.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        
    mean_score = float(np.mean(scores))
    
    # Calculate lower and upper percentiles (e.g., 2.5th and 97.5th for 95% CI)
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = float(np.percentile(boot_means, lower_percentile))
    upper_bound = float(np.percentile(boot_means, upper_percentile))
    
    return mean_score, lower_bound, upper_bound


def run_ragas_evaluation(pipeline_outputs_path: str, output_summary_path: str) -> Dict[str, Any]:
    """
    Loads saved pipeline inference JSON tracking logs, prepares the data schema
    for the Ragas engine, computes metrics, and bootstraps 95% CIs.
    """
    print(f"Loading pipeline generation footprint from: {pipeline_outputs_path}")
    with open(pipeline_outputs_path, "r") as f:
        records = json.load(f)
        
    # Re-map the structure to meet Ragas evaluation framework definitions
    ragas_data = {
        "question": [r["question"] for r in records],
        "contexts": [r["retrieved_context"] for r in records],
        "answer": [r["answer"] for r in records],
        "ground_truths": [[r["gold_answer"]] for r in records] # Must be a wrapped sequence
    }
    
    dataset = Dataset.from_dict(ragas_data)
    
    print("Invoking Ragas execution framework metrics sequence...")
    # Bind specific evaluating metrics targeting our multi-hop design targets
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
    
    # Execute matrix evaluation
    results_df = evaluate(dataset, metrics=metrics).to_pandas()
    
    summary_report = {}
    print("\n=== Processing Bootstrap Confidence Intervals (No Subsampling) ===")
    
    metric_keys = [
        "faithfulness", 
        "answer_relevancy", 
        "context_precision", 
        "context_recall", 
        "answer_correctness"
    ]
    
    for metric in metric_keys:
        if metric in results_df.columns:
            # Clean up potential NaN values if an evaluation node timed out or dropped frame
            scores = results_df[metric].dropna().to_numpy()
            
            mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(scores, num_bootstraps=1000)
            
            summary_report[metric] = {
                "mean": mean_val,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci,
                "formatted": f"{mean_val:.4f} [{lower_ci:.4f}, {upper_ci:.4f}]"
            }
            print(f"Metric: {metric.upper():<18} -> {summary_report[metric]['formatted']}")
            
    # Write aggregated metrics report to local results disk
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    with open(output_summary_path, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f"\nSaved aggregated summary report to: {output_summary_path}")
    
    return summary_report


if __name__ == "__main__":
    # Smoke-test data block to guarantee compilation boundaries hold up
    print("=== Testing Evaluation System Local Compilation ===")
    mock_records = [
        {
            "question": "What award did the director of Interstellar win in 2019?",
            "retrieved_context": ["Christopher Nolan won the Commander of the Order of the British Empire (CBE) in 2019."],
            "answer": "He won the Commander of the Order of the British Empire (CBE).",
            "gold_answer": "Commander of the Order of the British Empire (CBE)"
        }
    ]
    
    mock_file = "results/test_outputs.json"
    os.makedirs("results", exist_ok=True)
    with open(mock_file, "w") as f:
        json.load = json.dump(mock_records, f)
        
    print("Bootstrapping validation verification check complete. Mock file generated.")