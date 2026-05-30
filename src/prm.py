import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import RERANKER_MODEL

class ProcessRewardModel:
    def __init__(self, threshold: float = 0.4):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.threshold = threshold

    def score_step(self, context_query: str, evidence_text: str) -> float:
        # Standard cross-encoder format text construction
        input_text = f"{context_query} [SEP] {evidence_text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            logits = self.model(**inputs).logits.item()
            
        # Map raw unbounded logits to a deterministic 0.0 - 1.0 probability range
        probability = 1.0 / (1.0 + np.exp(-logits))
        return float(probability)

    def verify_step(self, score: float) -> bool:
        # Gate operation ensuring step meets the threshold criteria 
        return score >= self.threshold