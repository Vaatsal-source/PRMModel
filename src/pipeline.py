import os
import gc
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
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
        
        # Load lightweight synthesizer on CPU to meet the "synthesize answer" requirement 
        # without spiking VRAM.
        self.synthesizer = pipeline("text2text-generation", model="google/flan-t5-base", device="cpu")
        
        print(f"Loading PRM Scorer (DeBERTa) onto {self.device}...")
        self.prm_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.prm_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", 
            num_labels=1
        )
        
        if os.path.exists(prm_checkpoint_path):
            state_dict = torch.load(prm_checkpoint_path, map_location=self.device)
            self.prm_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded PRM weights from {prm_checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {prm_checkpoint_path}. Using base weights.")
            
        self.prm_model.to(self.device)
        self.prm_model.eval()
        
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
                score = torch.sigmoid(outputs.logits).squeeze().item()
                scores.append(score)
                
            del encoding
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
                
        return scores

    def run_pipeline(self, question: str, context_pool: List[str]) -> Dict[str, Any]:
        """
        Executes sequential multi-hop decomposition, local indexing, and PRM verification.
        """
        sub_questions = self.controller.decompose(question)
        self.retriever.build_local_index(context_pool)
        
        hop1_context = ""
        hop1_retrieved: List[str] = []
        
        if len(sub_questions) > 0:
            hop1_matches = self.retriever.iterative_retrieve(sub_questions[0], top_k=5)
            hop1_paragraphs = [match[0] for match in hop1_matches]
            hop1_scores = self._score_paragraphs_with_prm(sub_questions[0], hop1_paragraphs)
            
            for p, score in zip(hop1_paragraphs, hop1_scores):
                if score >= self.threshold:
                    hop1_retrieved.append(p)
            
            if not hop1_retrieved and hop1_paragraphs:
                hop1_retrieved = [hop1_paragraphs[0]]
                
            hop1_context = " ".join(hop1_retrieved)

        final_answer = "No clear answer could be formulated."
        best_hop2_context = ""
        
        if len(sub_questions) > 1:
            augmented_query = f"{sub_questions[1]} Context: {hop1_context}"
            hop2_matches = self.retriever.iterative_retrieve(augmented_query, top_k=3)
            hop2_paragraphs = [match[0] for match in hop2_matches]
            hop2_scores = self._score_paragraphs_with_prm(sub_questions[1], hop2_paragraphs)
            
            if hop2_scores:
                best_idx = hop2_scores.index(max(hop2_scores))
                best_hop2_context = hop2_paragraphs[best_idx]
                
                # Synthesize the final answer using the filtered context
                synth_prompt = f"Answer the question strictly based on the context.\nContext: {hop1_context} {best_hop2_context}\nQuestion: {question}\nAnswer:"
                final_answer = self.synthesizer(synth_prompt, max_new_tokens=64)[0]['generated_text']
                
        return {
            "decomposed_steps": sub_questions,
            "hop1_collected": hop1_retrieved,
            "answer": final_answer
        }