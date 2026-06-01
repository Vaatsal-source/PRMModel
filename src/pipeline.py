import os
import gc
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.retriever import HopController, LocalFAISSRetriever

class MultiHopRAGPipeline:
    """
    Coordinates multi-hop question answering by decomposing questions,
    retrieving context iteratively, and scoring candidates with a PRM.
    """
    def __init__(self, prm_checkpoint_path: str, threshold: float = 0.4, device: str = "cuda"):
        self.device = device
        self.threshold = threshold
        
        print(f"Loading Hop Components onto CPU...")
        self.controller = HopController()
        self.retriever = LocalFAISSRetriever()
        
        print(f"Loading PRM Scorer (DeBERTa) onto {self.device}...")
        # Using deberta-v3-base as the foundational architecture for the PRM
        self.prm_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.prm_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", 
            num_labels=1
        )
        
        # Load your custom trained PRM weights safely
        if os.path.exists(prm_checkpoint_path):
            state_dict = torch.load(prm_checkpoint_path, map_location=self.device)
            self.prm_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded PRM weights from {prm_checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {prm_checkpoint_path}. Using base weights.")
            
        self.prm_model.to(self.device)
        self.prm_model.eval()
        
        # Enable gradient checkpointing to aggressively preserve VRAM overhead
        if hasattr(self.prm_model, "gradient_checkpointing_enable"):
            self.prm_model.gradient_checkpointing_enable()

    def _score_paragraphs_with_prm(self, sub_question: str, paragraphs: List[str]) -> List[float]:
        """
        Scores paragraphs strictly one-by-one to eliminate VRAM spikes and fragmentation.
        """
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
                outputs = self.prm_model(encoding["input_ids"], encoding["attention_mask"])
                # Extract scalar logits directly
                score = torch.sigmoid(outputs.logits).squeeze().item()
                scores.append(score)
                
            # Evacuate intermediate tensor footprints immediately
            del encoding
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
                
        return scores

    def run_pipeline(self, question: str, context_pool: List[str]) -> Dict[str, Any]:
        """
        Executes sequential multi-hop decomposition, local indexing, and PRM verification.
        """
        # Step 1: Algorithmic decomposition via Flan-T5
        sub_questions = self.controller.decompose(question)
        
        # Step 2: Build transient local vector space
        self.retriever.build_local_index(context_pool)
        
        hop1_context = ""
        hop1_retrieved: List[str] = []
        
        # Hop 1 Execution Loop
        if len(sub_questions) > 0:
            hop1_matches = self.retriever.iterative_retrieve(sub_questions[0], top_k=5)
            hop1_paragraphs = [match[0] for match in hop1_matches]
            
            # Score Hop 1 candidates via the PRM Model
            hop1_scores = self._score_paragraphs_with_prm(sub_questions[0], hop1_paragraphs)
            
            # Filter candidates clearing our step validation threshold
            for p, score in zip(hop1_paragraphs, hop1_scores):
                if score >= self.threshold:
                    hop1_retrieved.append(p)
            
            # Fallback if no candidate meets the filter threshold
            if not hop1_retrieved and hop1_paragraphs:
                hop1_retrieved = [hop1_paragraphs[0]]
                
            hop1_context = " ".join(hop1_retrieved)

        # Hop 2 Execution Loop (Conditioned on Hop 1 insights)
        final_answer = "No clear answer could be formulated."
        if len(sub_questions) > 1:
            augmented_query = f"{sub_questions[1]} Context: {hop1_context}"
            hop2_matches = self.retriever.iterative_retrieve(augmented_query, top_k=3)
            hop2_paragraphs = [match[0] for match in hop2_matches]
            
            hop2_scores = self._score_paragraphs_with_prm(sub_questions[1], hop2_paragraphs)
            
            # Select the highest-scoring candidate path as the final anchor
            if hop2_scores:
                best_idx = hop2_scores.index(max(hop2_scores))
                final_answer = hop2_paragraphs[best_idx]
                
        return {
            "decomposed_steps": sub_questions,
            "hop1_collected": hop1_retrieved,
            "answer": final_answer
        }