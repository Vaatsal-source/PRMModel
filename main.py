from src.retriever import HotpotRetriever
from src.utils import set_seed
from src.config import SEED


def main():

    set_seed(SEED)

    retriever = HotpotRetriever()

    retriever.load_hotpotqa()

    retriever.build_chunks()

    retriever.build_faiss_index()

    query = (
        "The author of Harry Potter "
        "was born in which country?"
    )

    results = retriever.retrieve(query)

    print("\n" + "=" * 60)

    print(f"QUERY: {query}")

    print("=" * 60)

    for i, result in enumerate(results):

        print(f"\nRESULT {i+1}")

        print(f"TITLE: {result['title']}")

        print(f"TEXT: {result['text'][:300]}")

        print(f"DISTANCE: {result['distance']}")


if __name__ == "__main__":
    main()