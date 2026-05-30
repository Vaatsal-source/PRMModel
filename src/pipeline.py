from src.decomposer import QuestionDecomposer
from src.retriever import HotpotRetriever
from src.prm import ProcessRewardModel
from src.answer_generator import AnswerGenerator
from src.config import TOP_K

class MultiHopQAPipeline:
    def __init__(self, chunks=None, threshold: float = 0.4):
        self.decomposer = QuestionDecomposer()
        self.retriever = HotpotRetriever(chunks)
        self.prm = ProcessRewardModel(threshold=threshold)
        self.generator = AnswerGenerator()

    def run_pipeline(self, question: str) -> dict:
        sub_questions = self.decomposer.decompose(question)
        collected_evidence = []
        reasoning_trace = []
        accumulated_context = ""

        for hop_idx, sub_q in enumerate(sub_questions):
            # Create a true multi-hop link by appending accumulated context to sub-query 2
            search_query = f"{sub_q} {accumulated_context}".strip() if hop_idx > 0 else sub_q
            
            # 1. Fetch raw candidates
            candidates = self.retriever.retrieve(search_query, top_k=TOP_K)
            
            # 2. Score intermediate reasoning candidate steps using the PRM
            scored_candidates = []
            for doc in candidates:
                prm_prob = self.prm.score_step(sub_q, doc["text"])
                doc["prm_score"] = prm_prob
                scored_candidates.append(doc)
            
            # 3. Apply the threshold ablation gate 
            valid_candidates = [d for d in scored_candidates if self.prm.verify_step(d["prm_score"])]
            
            # Determine best step context forward
            if valid_candidates:
                best_chunk = max(valid_candidates, key=lambda x: x["prm_score"])
                is_pruned = False
            else:
                # If no steps pass the gate threshold, fallback to best max candidate score for robustness
                best_chunk = max(scored_candidates, key=lambda x: x["prm_score"]) if scored_candidates else None
                is_pruned = True
                
            if best_chunk:
                collected_evidence.append(best_chunk)
                accumulated_context += f" {best_chunk['text']}"
                
                reasoning_trace.append({
                    "hop": hop_idx + 1,
                    "sub_question": sub_q,
                    "resolved_query": search_query,
                    "selected_title": best_chunk.get("title", ""),
                    "prm_score": best_chunk["prm_score"],
                    "gate_pruned_fallback": is_pruned
                })

        # 4. Generate final answer output
        final_answer = self.generator.generate_answer(question, collected_evidence)
        
        return {
            "question": question,
            "predicted_answer": final_answer,
            "evidence": collected_evidence,
            "reasoning_trace": reasoning_trace
        }