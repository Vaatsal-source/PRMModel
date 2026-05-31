import torch
import faiss
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Enforce strict reproducibility across the board
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class HopController:
    """
    Handles question decomposition using google/flan-t5-large to break 
    multi-hop questions down into sequential single-hop search tasks.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Using bfloat16 for Flan-T5 on CUDA to respect VRAM constraints while maintaining numeric stability
        if "cuda" in str(device):
            current_dtype = torch.bfloat16 if torch.cuda.is_is_bf16_supported() else torch.float16
        else:
            current_dtype = torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, 
            torch_dtype=current_dtype
        ).to(self.device)
        self.model.eval()

    def decompose(self, question: str) -> List[str]:
        """
        Decomposes a complex question into sub-questions.
        Returns a list of sub-questions for sequential execution.
        """
        prompt = (
            f"Decompose the following multi-hop question into two distinct, sequential sub-questions "
            f"that need to be answered to solve the main question.\n"
            f"Question: {question}\n"
            f"Sub-questions (one per line, be concise):"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.0,  # Greedy decoding for absolute stability
                do_sample=False
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Split lines, strip artifacts, and ensure we filter out empty outputs
        sub_qs = [line.strip("- ").strip() for line in prediction.split("\n") if line.strip()]
        
        # Fallback guard: if decomposition fails to output 2 steps, create an algorithmic fallback
        if len(sub_qs) < 2:
            sub_qs = [question, question]
        return sub_qs[:2]


class LocalFAISSRetriever:
    """
    Manages a localized, transient FAISS index built strictly from the 10 
    candidate paragraphs provided in the HotpotQA distractor setting packet.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # BGE-base-en-v1.5 outputs 768-dimensional dense vectors
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
        self.dimension = 768
        self.paragraphs: List[str] = []
        self.index = None

    def build_local_index(self, paragraphs: List[str]):
        """
        Populates a completely fresh, standalone vector index for a single question context.
        """
        self.paragraphs = paragraphs
        # Compute sentence embeddings using standard normalization for inner product / L2 equivalency
        embeddings = self.embedder.encode(
            paragraphs, 
            batch_size=len(paragraphs), 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)
        
        # Using IndexFlatIP (Inner Product) since our vectors are normalized
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def iterative_retrieve(self, sub_query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Queries the localized 10-paragraph pool using a single sub-question.
        Returns a list of tuples: (paragraph_text, confidence_score)
        """
        if self.index is None or not self.paragraphs:
            raise ValueError("Local FAISS index has not been initialized with candidate paragraphs.")
            
        query_vector = self.embedder.encode([sub_query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.paragraphs):  # Safeguard against FAISS out-of-bounds indicators
                results.append((self.paragraphs[idx], float(score)))
        return results


# Self-contained testing harness to run validation checks directly
if __name__ == "__main__":
    print("=== Testing Retriever Stack Locally ===")
    
    # 1. Initialize models
    print("Initializing components...")
    controller = HopController(device="cpu")  # Using CPU locally for quick verification
    retriever = LocalFAISSRetriever(device="cpu")
    
    # 2. Mock HotpotQA instance (1 Question, 10 Localized Paragraphs)
    mock_question = "Which film directed by Christopher Nolan stars the actor who played Arthur in Inception?"
    mock_paragraphs = [
        "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
        "Joseph Gordon-Levitt portrayed Arthur in Christopher Nolan's 2010 film Inception.",
        "The Dark Knight Rises is a 2012 superhero film directed by Christopher Nolan, starring Joseph Gordon-Levitt as Blake.",
        "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan.",
        "Dunkirk is a 2017 war film written, directed, and co-produced by Christopher Nolan.",
        "Memento is a 2000 American psychological thriller film written and directed by Christopher Nolan.",
        "The Prestige is a 2006 psychological thriller film directed by Christopher Nolan.",
        "Following is a 1998 neo-noir crime thriller film written and directed by Christopher Nolan.",
        "Tenet is a 2020 science fiction action thriller film written and directed by Christopher Nolan.",
        "Insomnia is a 2002 American psychological thriller film directed by Christopher Nolan."
    ]
    
    # 3. Verify sub-question breakdown
    print(f"\nTesting Decomposition for: '{mock_question}'")
    sub_questions = controller.decompose(mock_question)
    for i, sub_q in enumerate(sub_questions, 1):
        print(f"   Hop {i} Sub-Question: {sub_q}")
        
    # 4. Verify Local Vector Lookups
    print("\nBuilding Transient local FAISS index...")
    retriever.build_local_index(mock_paragraphs)
    
    print(f"Querying Hop 1 target: '{sub_questions[0]}'")
    hop1_matches = retriever.iterative_retrieve(sub_questions[0], top_k=2)
    for txt, score in hop1_matches:
        print(f"   [Score: {score:.4f}] -> {txt[:80]}...")