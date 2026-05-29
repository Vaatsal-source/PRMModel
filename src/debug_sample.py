from datasets import load_dataset

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

sample = dataset["validation"][0]

print(sample.keys())