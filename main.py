from src.pipeline import MultiHopQAPipeline
from src.utils import set_seed
from src.config import SEED


def main():

    set_seed(SEED)

    pipeline = MultiHopQAPipeline()

    query = (
        "The author of Harry Potter "
        "was born in which country?"
    )

    output = pipeline.answer_question(query)

    print("\n" + "=" * 60)

    print("FINAL OUTPUT")

    print("=" * 60)

    for step in output["reasoning_steps"]:

        print(f"\nHOP {step['hop']}")

        print(f"QUERY: {step['query']}")

        for i, retrieved in enumerate(
            step["retrieved"]
        ):

            print(f"\nRESULT {i+1}")

            print(
                f"TITLE: "
                f"{retrieved['title']}"
            )

            print(
                f"TEXT: "
                f"{retrieved['text'][:300]}"
            )


if __name__ == "__main__":
    main()