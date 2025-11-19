from datasets import load_dataset, concatenate_datasets

wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

wikipedia.save_to_disk("dataset")