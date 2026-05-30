import torch
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.retriever import HopController, LocalFAISSRetriever
from src.prm import PRMScorer, threshold_prune

# Enforce strict reproducibility across runtime boundaries
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class MultiHopRAGPipeline:
    """
    Master coordination pipeline executing multi-hop decomposition,
    local context retrieval, PRM pruning, and final answer generation.
    """
    def __init__(self, prm_checkpoint_path: str = None, threshold: float = 0.4, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.threshold = threshold
        
        print("Initializing Pipeline Core Components...")
        # 1. Initialize retriever components
        self.hop_controller = HopController(device=self.device)
        self.local_retriever = LocalFAISSRetriever(device=self.device)
        
        # 2. Initialize and load PRM Cross-Encoder
        self.prm_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.prm_model = PRMScorer().to(self.device)
        if prm_checkpoint_path and torch.cuda.is_available():
            self.prm_model.load_state_dict(torch.load(prm_checkpoint_path, map_location=self.device))
            print(f"Loaded tuned PRM weights from: {prm_checkpoint_path}")
        self.prm_model.eval()

        # 3. Share Flan-T5 model instance from hop_controller for final generation to conserve VRAM
        self.gen_model = self.hop_controller.model
        self.gen_tokenizer = self.hop_controller.tokenizer

    def _score_paragraphs_with_prm(self, sub_question: str, paragraphs: List[str]) -> List[float]:
        """
        Passes (Sub-Question, Paragraph) pairs through the DeBERTa Cross-Encoder.
        """
        if not paragraphs:
            return []
            
        scores = []
        for p in paragraphs:
            encoding = self.prm_tokenizer(
                sub_question,
                p,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                score = self.prm_model(encoding["input_ids"], encoding["attention_mask"])
                scores.append(score.item())
        return scores

    def _generate_final_answer(self, question: str, reasoning_context: List[str]) -> str:
        """
        Uses Flan-T5-large to generate a concise answer over the pruned reasoning chain context.
        """
        joined_context = "\n".join([f"- {c}" for c in reasoning_context])
        prompt = (
            f"Context:\n{joined_context}\n\n"
            f"Based on the provided context above, answer the following question clearly and concisely.\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gen_model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0, # Greedy decoding for absolute deterministic consistency
                do_sample=False
            )
        return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_pipeline(self, question: str, candidate_paragraphs: List[str]) -> Dict[str, Any]:
        """
        Executes the entire RAG pipeline for a single question instance.
        """
        # Step 1: Decompose original question into multi-hop sub-questions
        sub_questions = self.hop_controller.decompose(question)
        
        # Step 2: Spin up local transient FAISS index for the question's specific 10 paragraphs
        self.local_retriever.build_local_index(candidate_paragraphs)
        
        accumulated_context = []
        
        # Step 3: Run Multi-Hop retrieval loops
        for hop_idx, sub_q in enumerate(sub_questions):
            # Retrieve top 4 items locally to give PRM a rich set to filter from
            retrieved_tuples = self.local_retriever.iterative_retrieve(sub_q, top_k=4)
            retrieved_texts = [text for text, _ in retrieved_tuples]
            
            # Score retrieved options with the PRM Cross-Encoder
            prm_scores = self._score_paragraphs_with_prm(sub_q, retrieved_texts)
            
            # Prune out distractors matching our threshold rules (t=0.4 or t=0.6)
            pruned_hop_context = threshold_prune(retrieved_texts, prm_scores, t=self.threshold)
            
            for text in pruned_hop_context:
                if text not in accumulated_context:
                    accumulated_context.append(text)
                    
        # Step 4: Synthesize answer over clean, high-confidence evidence
        final_answer = self._generate_final_answer(question, accumulated_context)
        
        return {
            "question": question,
            "sub_questions": sub_questions,
            "retrieved_context": accumulated_context,
            "answer": final_answer
        }


def load_held_out_evaluation_split(num_samples: int = 500) -> List[Dict[str, Any]]:
    """
    Loads and prepares the 500 deterministic evaluation items from the 
    official HotpotQA validation dataset configuration.
    """
    print("Loading official HotpotQA validation split...")
    raw_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Deterministic shuffle to lock the exact same evaluation footprint across runs
    shuffled_dataset = raw_dataset.shuffle(seed=SEED)
    
    evaluation_packet = []
    for i in range(min(num_samples, len(shuffled_dataset))):
        item = shuffled_dataset[i]
        
        # Reconstruct full continuous strings from the HotpotQA nested context arrays
        paragraphs = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            paragraph_text = f"{title}: " + "".join(sentences)
            paragraphs.append(paragraph_text)
            
        evaluation_packet.append({
            "question": item["question"],
            "paragraphs": paragraphs,
            "gold_answer": item["answer"]
        })
        
    return evaluation_packet


if __name__ == "__main__":
    print("=== Testing Integration Pipeline Locally ===")
    
    # Spin up pipeline running completely on CPU for rapid integration validation check
    pipeline = MultiHopRAGPipeline(prm_checkpoint_path=None, threshold=0.4, device="cpu")
    
    mock_q = "What award did the director of Interstellar win in 2019?"
    mock_10_paras = [
        "Interstellar: Interstellar is an epic science fiction film directed by Christopher Nolan.",
        "Christopher Nolan: Christopher Nolan won the Commander of the Order of the British Empire (CBE) in 2019.",
        "Distractor 1: Random irrelevant sentence about sports data tracking metrics.",
        "Distractor 2: Weather patterns in Western Europe changed dramatically during the late summer months.",
        "Distractor 3: Deep learning pipelines run highly efficiently when vector stores are properly aligned.",
        "Distractor 4: Coffee consumption worldwide reached an all time peak during late 2024 studies.",
        "Distractor 5: Stock market options require careful management of risk parameters.",
        "Distractor 6: Python 3.11 introduces faster specialized bytecode interpreter loops.",
        "Distractor 7: Electric vehicles utilize lithium-ion cells for energy dense applications.",
        "Distractor 8: The Great Barrier Reef contains highly diverse marine ecosystem structures."
    ]
    
    print("\nRunning mock instance execution tracking loop...")
    result = pipeline.run_pipeline(mock_q, mock_10_paras)
    
    print("\n[Pipeline Results Output]")
    print(f"Decomposed Steps: {result['sub_questions']}")
    print(f"Pruned Context Chunks Retained: {len(result['retrieved_context'])}")
    print(f"Generated Answer Output: {result['answer']}")