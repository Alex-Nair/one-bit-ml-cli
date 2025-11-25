from datasets import load_from_disk
import sentencepiece as spm

dataset = load_from_disk("../../../dataset/dataset")

def text_iterator():
    for item in dataset:
        yield item["text"]

if __name__ == "__main__":
    spm.SentencePieceTrainer.train(
        sentence_iterator=text_iterator(),
        model_prefix="tokenizer",
        vocab_size=32000,
        character_coverage=1.0,
        model_type="bpe"
    )