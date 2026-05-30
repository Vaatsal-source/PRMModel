import json
from src.config import DATA_DIR
from src.retriever import HotpotRetriever

def build_global_corpus():
    heldout_path = DATA_DIR / "heldout_500.json"
    if not heldout_path.exists():
        raise FileNotFoundError(
            f"Could not find {heldout_path}. Please ensure your dataset extraction script saves it to the data/ folder."
        )

    with open(heldout_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Flatten out context paragraphs to create a single un-duplicated corpus
    seen_titles = set()
    flat_chunks = []

    print("[INFO] Processing dataset paragraphs...")
    for sample in samples:
        context_data = sample.get("context", {})
        titles = context_data.get("title", [])
        sentences_list = context_data.get("sentences", [])

        for title, sentences in zip(titles, sentences_list):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Combine sentences list into a continuous block of paragraph text
            paragraph_text = "".join(sentences)
            flat_chunks.append({
                "title": title,
                "text": paragraph_text
            })

    print(f"[INFO] Total unique context paragraphs extracted: {len(flat_chunks)}")
    
    # Initialize retriever and write to disk
    retriever = HotpotRetriever(chunks=flat_chunks)
    retriever.build_index()

if __name__ == "__main__":
    build_global_corpus()