import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Enforce strict reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class PRMDataset(Dataset):
    """
    Dataset wrapper for tokenizing sub-question and paragraph pairs on the fly.
    """
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data  # List of dicts containing {"question": str, "passage": str, "label": int}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["question"],
            item["passage"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }


class PRMScorer(nn.Module):
    """
    DeBERTa-v3-base Cross-Encoder that outputs raw logits for training numerical stability,
    with an internal sigmoid method for clean probability extraction.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Binary classification head processing the pooled representation
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits.squeeze(-1) # Output raw un-activated logits (Safe for AMP Autocast)


def train_prm(
    train_data: List[Dict], 
    val_data: List[Dict], 
    output_dir: str, 
    epochs: int = 3, 
    batch_size: int = 8, 
    lr: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Fine-tuning sequence for training the PRM cross-encoder on Kaggle using BCEWithLogitsLoss.
    """
    print(f"Starting PRM Training on device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = PRMScorer().to(device)
    
    train_dataset = PRMDataset(train_data, tokenizer)
    val_dataset = PRMDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # BCEWithLogitsLoss natively handles stability inside mixed-precision contexts
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    
    use_cuda = "cuda" in str(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            total_train_loss += loss.item()
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Train Loss: {loss.item():.4f}")
                
        # Validation Pass
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    logits = model(input_ids, attention_mask)
                    val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"--- Epoch {epoch+1} Complete | Avg Val Loss: {avg_val_loss:.4f} ---")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "prm_deberta_best.pt"))
            print(f"Saved new best model checkpoint to {output_dir}")


def threshold_prune(passages: List[str], scores: List[float], t: float) -> List[str]:
    """
    Applies the PRM gate threshold ablation rule. 
    Filters out any context blocks that score strictly below the target threshold value.
    """
    pruned_passages = [p for p, score in zip(passages, scores) if score >= t]
    
    # Algorithmic Safety Fallback: If everything gets pruned, preserve at least the highest scoring chunk
    if not pruned_passages and passages:
        max_idx = np.argmax(scores)
        pruned_passages.append(passages[max_idx])
        
    return pruned_passages


if __name__ == "__main__":
    print("=== Testing PRM Module Logic Locally ===")
    
    # Test pruning functionality
    mock_passages = ["Passage A", "Passage B", "Passage C"]
    mock_scores = [0.25, 0.55, 0.85]
    
    print("Testing Ablation Threshold t=0.4:")
    print("   Filtered:", threshold_prune(mock_passages, mock_scores, t=0.4))
    
    print("Testing Ablation Threshold t=0.6:")
    print("   Filtered:", threshold_prune(mock_passages, mock_scores, t=0.6))
    
    print("Testing Safety Fallback (all scores low, t=0.6):")
    print("   Filtered:", threshold_prune(mock_passages, [0.1, 0.2, 0.15], t=0.6))