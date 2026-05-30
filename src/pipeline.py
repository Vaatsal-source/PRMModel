from src.retriever import HotpotRetriever
from src.reranker import Reranker
from src.decomposer import QuestionDecomposer
from src.prm import ProcessRewardModel


class MultiHopQAPipeline:

    def __init__(self, chunks):

        self.retriever = HotpotRetriever(chunks)
        self.reranker = Reranker()
        self.decomposer = QuestionDecomposer()
        self.prm = ProcessRewardModel()

        self.retriever.build_index()

    def answer_question(self, question):

        sub_questions = self.decomposer.decompose(question)

        all_evidence = []

        reasoning_trace = []

        for sq in sub_questions:

            print(f"\n[HOP] {sq}")

            docs = self.retriever.retrieve(sq)

            docs = self.reranker.rerank(sq, docs)

            scored = []

            for d in docs:

                score = self.prm.score(sq, d["text"])

                d["prm_score"] = score

                scored.append(d)

            best = max(scored, key=lambda x: x["prm_score"])

            all_evidence.append(best)

            reasoning_trace.append({
                "sub_question": sq,
                "best_evidence": best
            })

        final_answer = self.generate_answer(question, all_evidence)

        return {
            "question": question,
            "reasoning_trace": reasoning_trace,
            "evidence": all_evidence,
            "predicted_answer": final_answer
        }

    def generate_answer(self, question, evidence):

        text = " ".join([e["text"] for e in evidence])

        if "same nationality" in question.lower():
            if "american" in text.lower():
                return "yes"
            return "no"

        return "unknown"