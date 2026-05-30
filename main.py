from datasets import load_dataset

from src.retriever import (
    HotpotRetriever
)

from src.reranker import (
    CrossEncoderReranker
)

from src.utils import set_seed

from src.config import SEED


def main():

    set_seed(SEED)

    print(
        "[INFO] Loading HotpotQA..."
    )

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor"
    )

    sample = dataset[
        "validation"
    ][0]

    question = sample["question"]

    print("\n" + "="*80)
    print("QUESTION")
    print("="*80)
    print(question)

    print("\n" + "="*80)
    print("GOLD ANSWER")
    print("="*80)
    print(sample["answer"])

    # =====================================
    # RETRIEVER
    # =====================================

    retriever = (
        HotpotRetriever()
    )

    docs = retriever.build_docs(
        sample
    )

    retrieved = (
        retriever.retrieve(
            question,
            docs
        )
    )

    print("\n" + "="*80)
    print("RETRIEVAL RESULTS")
    print("="*80)

    for idx, doc in enumerate(
        retrieved[:5]
    ):

        print(
            f"\n{idx+1}. "
            f"{doc['title']}"
        )

        print(
            f"Score: "
            f"{doc['score']:.4f}"
        )

    # =====================================
    # RERANKER
    # =====================================

    reranker = (
        CrossEncoderReranker()
    )

    reranked = (
        reranker.rerank(
            question,
            retrieved
        )
    )

    print("\n" + "="*80)
    print("RERANKED RESULTS")
    print("="*80)

    for idx, doc in enumerate(
        reranked
    ):

        print(
            f"\n{idx+1}. "
            f"{doc['title']}"
        )

        print(
            f"Rerank Score: "
            f"{doc['rerank_score']:.4f}"
        )

        print(
            doc["text"][:250]
        )

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":

    main()