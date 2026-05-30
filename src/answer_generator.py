import torch
from transformers import pipeline
from src.config import QA_MODEL

class AnswerGenerator:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = pipeline(
            "question-answering",
            model=QA_MODEL,
            device=device
        )

    def generate_answer(self, question: str, evidence_list: list) -> str:
        if not evidence_list:
            return "Unknown"
            
        # Join extracted contextual documents into a single block
        combined_context = "\n".join([f"Document: {e['title']}\n{e['text']}" for e in evidence_list])
        
        # Guard rail against model max context limit truncation
        result = self.qa_pipeline(question=question, context=combined_context[:4000])
        return result["answer"].strip()