from sentence_transformers import CrossEncoder
import torch


class ProcessRewardModel:

    def __init__(
        self,
        threshold=0.4
    ):

        print("[INFO] Loading PRM model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.threshold = threshold

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=self.device
        )

    # =====================================
    # SCORE REASONING STEP
    # =====================================

    def score_step(
        self,
        question,
        reasoning_step
    ):

        score = self.model.predict(
            [
                (
                    question,
                    reasoning_step
                )
            ]
        )[0]


        return float(score)

    # =====================================
    # THRESHOLD PRUNING
    # =====================================

    def keep_step(
        self,
        score
    ):

        return score >= self.threshold

    # =====================================
    # UPDATE THRESHOLD
    # =====================================

    def set_threshold(
        self,
        threshold
    ):

        self.threshold = threshold