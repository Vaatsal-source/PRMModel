import sys, os, json, random, torch, gc
from datasets import load_dataset
from tqdm import tqdm
from src.pipeline import MultiHopRAGPipeline

# Set split configuration safely for the Colab GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()
gc.collect()

device_target = "cuda" if torch.cuda.is_available() else "cpu"
prm_checkpoint_path = "checkpoints/prm_deberta_best/prm_deberta_best.pt"
full_val_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
eval_dataset = full_val_dataset.select(range(500))

print(f"🚀 Initializing Pipeline on device: {device_target}...")
pipeline = MultiHopRAGPipeline(
    prm_checkpoint_path=prm_checkpoint_path, 
    threshold=0.4, 
    device=device_target
)

run_outputs = []
for item in tqdm(eval_dataset, desc="Evaluating t=0.4"):
    context_pool = [f"{t}: " + "".join(s) for t, s in zip(item["context"]["title"], item["context"]["sentences"])]
    
    res = pipeline.run_pipeline(item["question"], context_pool)
    run_outputs.append({"question": item["question"], "answer": res["answer"]})
    
    # Protect against any residual fragmentation
    if device_target == "cuda":
        torch.cuda.empty_cache()

# Ensure results folder exists locally
os.makedirs("results", exist_ok=True)
with open("results/pipeline_outputs_t04.json", "w") as f:
    json.dump(run_outputs, f, indent=4)
print("✅ Evaluation complete. Saved to results/pipeline_outputs_t04.json")