from src.retriever import HotpotRetriever
from src.hop_controller import HopController
from src.query_rewriter import QueryRewriter
from src.reranker import Reranker
from src.config import RETRIEVAL_K, FINAL_K


class MultiHopQAPipeline:

    def __init__(self):

        self.reranker = Reranker()

        self.query_rewriter = QueryRewriter()

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
                current_query,
                top_k=RETRIEVAL_K
            )

            retrieved = self.reranker.rerank(
                current_query,
                retrieved,
                top_k=FINAL_K
            )

            print("\n[INFO] Top reranked documents:")

            for i, doc in enumerate(retrieved[:3]):

                print(
                    f"{i+1}. "
                    f"{doc['title']} | "
                    f"Score={doc['rerank_score']:.4f}"
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
                self.query_rewriter
                .rewrite(
                    query,
                    bridge_entity
                )
            )

            print(
                f"[INFO] Next Query: "
                f"{current_query}"
            )

        return {
            "question": query,
            "reasoning_steps": reasoning_steps,
            "retrieved_context": all_retrieved_chunks
        }