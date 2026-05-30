from transformers import pipeline
import torch


class AnswerGenerator:

    def __init__(self):

        print("[INFO] Loading QA model...")

        device = 0 if torch.cuda.is_available() else -1

        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device
        )

    # =====================================
    # GENERATE ANSWER
    # =====================================

    def generate_answer(
        self,
        question,
        documents
    ):

        if len(documents) == 0:

            return "Unknown"

        context = "\n".join(
            doc["text"]
            for doc in documents
        )

        result = self.qa_pipeline(
            question=question,
            context=context[:4000]
        )

        return result["answer"]