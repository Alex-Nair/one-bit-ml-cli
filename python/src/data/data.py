from datasets import load_from_disk

def load_dataset():
    dataset = load_from_disk("../dataset/dataset")
    return dataset