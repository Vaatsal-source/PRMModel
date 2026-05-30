from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ProcessRewardModel:

    def __init__(self):

        print("[INFO] Loading PRM model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.model.eval()

        self.threshold = 5.0

    def score(self, question, evidence):

        text = question + " [SEP] " + evidence

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            score = self.model(**inputs).logits.item()

        return score