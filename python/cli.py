from src.llm.train import tokenize_dataset
from src.llm.parts import *

from src.data.data import load_dataset
import sentencepiece as spm
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    tokenizer = spm.SentencePieceProcessor(model_file = "src/tokenization/tokenizer.model")
    vocabSize = tokenizer.GetPieceSize()

    tokenize_dataset()