from src.retriever import HotpotRetriever
from src.hop_controller import HopController


class MultiHopQAPipeline:

    def __init__(self):

        self.retriever = HotpotRetriever()

        self.hop_controller = HopController()

        self.retriever.load_hotpotqa()

        self.retriever.build_chunks()

        self.retriever.build_faiss_index()

    # =====================================
    # RUN MULTI-HOP PIPELINE
    # =====================================

    def answer_question(
        self,
        query,
        hops=2
    ):

        reasoning_steps = []

        current_query = query

        all_retrieved_chunks = []

        for hop in range(hops):

            print(f"\n[INFO] Hop {hop+1}")

            retrieved = self.retriever.retrieve(
                current_query
            )

            all_retrieved_chunks.extend(
                retrieved
            )

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

            print(
                f"[INFO] Bridge Entity: "
                f"{bridge_entity}"
            )

            if bridge_entity is None:
                break

            current_query = (
                self.hop_controller
                .reformulate_query(
                    query,
                    bridge_entity
                )
            )

        return {
            "question": query,
            "reasoning_steps": reasoning_steps,
            "retrieved_context": all_retrieved_chunks
        }