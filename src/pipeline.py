from src.retriever import HotpotRetriever
from src.hop_controller import HopController
from src.query_rewriter import QueryRewriter


class MultiHopQAPipeline:

    def __init__(self):

        self.retriever = HotpotRetriever()

        self.hop_controller = HopController()

        self.query_rewriter = QueryRewriter()

    # =====================================
    # RUN MULTI-HOP
    # =====================================

    def answer_question(
        self,
        sample,
        hops=2
    ):

        question = sample["question"]

        docs = self.retriever.build_docs(
            sample
        )

        current_query = question

        reasoning_steps = []

        all_docs = []

        for hop in range(hops):

            print(f"\n[INFO] Hop {hop+1}")

            retrieved = self.retriever.retrieve(
                current_query,
                docs
            )

            all_docs.extend(retrieved)

            reasoning_steps.append({
                "hop": hop + 1,
                "query": current_query,
                "retrieved": retrieved
            })

            bridge_entity = (
                self.hop_controller
                .select_bridge_entity(
                    current_query,
                    retrieved
                )
            )

            if bridge_entity is None:

                break

            print(
                f"[INFO] Bridge Entity: "
                f"{bridge_entity}"
            )

            current_query = (
                self.query_rewriter
                .rewrite(
                    question,
                    bridge_entity
                )
            )

            print(
                f"[INFO] Next Query: "
                f"{current_query}"
            )

        return {
            "question": question,
            "answer": sample["answer"],
            "reasoning_steps": reasoning_steps,
            "retrieved_context": all_docs
        }