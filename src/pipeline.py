from src.retriever import HotpotRetriever
from src.prm import ProcessRewardModel
from src.hop_controller import HopController
from src.query_rewriter import QueryRewriter
from src.answer_generator import AnswerGenerator

class MultiHopQAPipeline:

    def __init__(self):

        self.answer_generator = AnswerGenerator()

        self.retriever = HotpotRetriever()

        self.prm = ProcessRewardModel()

        self.hop_controller = HopController()

        self.query_rewriter = QueryRewriter()

    # ==================================================
    # REASONING STEP CREATION
    # ==================================================

    def create_reasoning_step(
        self,
        question,
        document
    ):

        return (
            f"Question: {question}\n"
            f"Evidence: {document['text'][:400]}"
        )

    # ==================================================
    # MULTI-HOP PIPELINE
    # ==================================================

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

        reasoning_trace = []

        all_retrieved_docs = []

        all_kept_docs = []

        for hop in range(hops):

            print(f"\n{'='*60}")
            print(f"HOP {hop+1}")
            print(f"{'='*60}")

            # -----------------------------------------
            # RETRIEVAL
            # -----------------------------------------

            retrieved_docs = self.retriever.retrieve(
                current_query,
                docs
            )

            all_retrieved_docs.extend(
                retrieved_docs
            )

            kept_docs = []

            hop_reasoning = []

            print("\n[INFO] Scoring reasoning steps...\n")

            # -----------------------------------------
            # PRM SCORING
            # -----------------------------------------

            for doc in retrieved_docs:

                reasoning_step = (
                    self.create_reasoning_step(
                        current_query,
                        doc
                    )
                )

                score = self.prm.score_step(
                    current_query,
                    reasoning_step
                )

                hop_reasoning.append({
                    "title": doc["title"],
                    "score": score,
                    "text": doc["text"]
                })

                print(
                    f"{doc['title']:<40}"
                    f" PRM={score:.4f}"
                )

                if score >= self.prm.threshold:

                    kept_docs.append(doc)

            all_kept_docs.extend(
                kept_docs
            )

            reasoning_trace.append({
                "hop": hop + 1,
                "query": current_query,
                "steps": hop_reasoning
            })

            # -----------------------------------------
            # NO DOCUMENT SURVIVED
            # -----------------------------------------

            if len(kept_docs) == 0:

                print(
                    "\n[INFO] No document survived PRM."
                )

                break

            # -----------------------------------------
            # BRIDGE ENTITY SELECTION
            # -----------------------------------------

            bridge_entity = (
                self.hop_controller
                .select_bridge_entity(
                    current_query,
                    kept_docs
                )
            )

            print(
                f"\n[INFO] Bridge Entity: "
                f"{bridge_entity}"
            )

            if bridge_entity is None:

                break

            # -----------------------------------------
            # QUERY REWRITE
            # -----------------------------------------

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

            final_answer = (
                self.answer_generator.generate_answer(
                question,
                all_kept_docs
            )
        )

        return {

            "question": question,

            "gold_answer": sample["answer"],

            "predicted_answer": final_answer,

            "retrieved_docs": all_retrieved_docs,

            "kept_docs": all_kept_docs,

            "reasoning_trace": reasoning_trace
        }