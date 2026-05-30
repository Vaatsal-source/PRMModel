from datasets import load_dataset
from src.pipeline import MultiHopQAPipeline


def main():

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    sample = dataset["validation"][0]

    print("\nQUESTION:", sample["question"])
    print("ANSWER:", sample["answer"])

    chunks = []

    for t, sents in zip(
        sample["context"]["title"],
        sample["context"]["sentences"]
    ):
        chunks.append({
            "title": t,
            "text": " ".join(sents)
        })

    pipeline = MultiHopQAPipeline(chunks)

    output = pipeline.answer_question(sample["question"])

    print("\nFINAL ANSWER:", output["predicted_answer"])

    print("\nTRACE:")
    for step in output["reasoning_trace"]:
        print(step)


if __name__ == "__main__":
    main()