import os
import json
import torch
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

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

SEED = 42
np.random.seed(SEED)

def bootstrap_confidence_interval(
    scores: np.ndarray, 
    num_bootstraps: int = 1000, 
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
        
    boot_means = []
    n = len(scores)
    
    for _ in range(num_bootstraps):
        boot_sample = np.random.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        
    mean_score = float(np.mean(scores))
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = float(np.percentile(boot_means, lower_percentile))
    upper_bound = float(np.percentile(boot_means, upper_percentile))
    
    return mean_score, lower_bound, upper_bound

def get_opensource_evaluators():
    print("Initializing local Open-Source Evaluation Stack (Phi-3 & BGE)...")
    eval_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )
    
    base_llm = HuggingFacePipeline(pipeline=pipe)
    eval_llm = ChatHuggingFace(llm=base_llm)
    
    return eval_llm, eval_embeddings

def run_ragas_evaluation(pipeline_outputs_path: str, output_summary_path: str) -> Dict[str, Any]:
    print(f"Loading pipeline generation footprint from: {pipeline_outputs_path}")
    with open(pipeline_outputs_path, "r") as f:
        records = json.load(f)
        
    ragas_data = {
        "question": [r["question"] for r in records],
        "contexts": [r["retrieved_context"] for r in records],
        "answer": [r["answer"] for r in records],
        "ground_truths": [[r["gold_answer"]] for r in records] 
    }
    
    dataset = Dataset.from_dict(ragas_data)
    eval_llm, eval_embeddings = get_opensource_evaluators()
    
    print("Invoking Ragas execution framework metrics sequence...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
    
    results_df = evaluate(
        dataset, 
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings
    ).to_pandas()
    
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
            scores = results_df[metric].dropna().to_numpy()
            mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(scores, num_bootstraps=1000)
            
            summary_report[metric] = {
                "mean": mean_val,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci,
                "formatted": f"{mean_val:.4f} [{lower_ci:.4f}, {upper_ci:.4f}]"
            }
            print(f"Metric: {metric.upper():<18} -> {summary_report[metric]['formatted']}")
            
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    
    # Required: Output both JSON and CSV
    with open(output_summary_path, "w") as f:
        json.dump(summary_report, f, indent=4)
        
    csv_path = output_summary_path.replace(".json", ".csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nSaved aggregated summary report to: {output_summary_path}")
    print(f"Saved raw execution metrics to: {csv_path}")
    
    return summary_report

if __name__ == "__main__":
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
        json.dump(mock_records, f)
        
    # Will generate both test_summary.json and test_summary.csv
    run_ragas_evaluation(mock_file, "results/test_summary.json")
    print("Bootstrapping validation verification check complete. Mock files generated.")