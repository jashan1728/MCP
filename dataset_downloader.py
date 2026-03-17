from datasets import load_dataset

dataset = load_dataset(
    "shunk031/MSCOCO",
    year=2014,
    coco_task="captions"
)

print(dataset)
print(dataset["train"][0])