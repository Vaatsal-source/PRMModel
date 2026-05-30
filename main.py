from datasets import load_dataset

from src.pipeline import MultiHopQAPipeline
from src.utils import set_seed
from src.config import SEED


def main():

    set_seed(SEED)

    print("[INFO] Loading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor"
    )

    sample = dataset["validation"][0]

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(sample["question"])

    print("\n" + "=" * 80)
    print("GOLD ANSWER")
    print("=" * 80)
    print(sample["answer"])

    print("\n" + "=" * 80)
    print("AVAILABLE DOCUMENTS")
    print("=" * 80)

    for idx, title in enumerate(
        sample["context"]["title"]
    ):
        print(f"{idx+1}. {title}")

    pipeline = MultiHopQAPipeline()

    output = pipeline.answer_question(
        sample,
        hops=2
    )

    print("\n")
    print("=" * 80)
    print("PREDICTED ANSWER")
    print("=" * 80)
    print(output["predicted_answer"])

    print("\n")
    print("=" * 80)
    print("GOLD ANSWER")
    print("=" * 80)
    print(output["gold_answer"])

    print("\n")
    print("=" * 80)
    print("REASONING TRACE")
    print("=" * 80)

    for hop_data in output["reasoning_trace"]:

        print(f"\nHOP {hop_data['hop']}")
        print("-" * 60)

        for step in hop_data["steps"]:

            print(
                f"{step['title']:<40} "
                f"PRM={step['score']:.4f}"
            )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()