from datasets import load_dataset

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

sample = dataset["validation"][0]

print("\nQUESTION:")
print(sample["question"])

print("\nANSWER:")
print(sample["answer"])

print("\nSUPPORTING FACTS:")

titles = sample["supporting_facts"]["title"]

for t in titles:

    print("-", t)