from datasets import load_dataset
import numpy as np
import sentencepiece as spm
import time

def tokenize_dataset(batchSize = 256, batchLogAmount=1000):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming = True)
    tokenizer = spm.SentencePieceProcessor(model_file = "src/tokenization/tokenizer.model")

    # Log Info
    totalRows = 0
    totalTokens = 0
    startTime = time.time()
    logNumber = 1

    batch = []

    with open("tokens.bin", "wb") as f:
        for row in dataset:
            batch.append(row["text"])

            if len(batch) == batchSize:
                idsBatch = [tokenizer.encode(t, out_type = int) for t in batch]

                for ids in idsBatch:
                    totalTokens += len(ids)
                    np.array(ids, dtype = np.uint32).tofile(f)
                
                totalRows += batchSize
                batch = []

                if totalRows % (batchSize * batchLogAmount) == 0:
                    elapsed = time.time() - startTime
                    gbSpace = (totalTokens * 4) / (1024 ** 3)

                    print(f"Log #{logNumber}:")
                    print(f"Total Rows: {totalRows}")
                    print(f"Total Tokens: {totalTokens}")
                    print(f"GBs Written: {gbSpace:.2f}")
                    print(f"Hours Elapsed: {elapsed / 3600:.2f}\n\n")

                    logNumber += 1

        # Leftover rows
        if batch:
            idsBatch = [tokenizer.encode(t, out_type = int) for t in batch]

            for ids in idsBatch:
                totalTokens += len(ids)
                np.array(ids, dtype = np.uint32).tofile(f)
            
            totalRows += batchSize
    
    # Last Log
    print(f"Tokenization Completed. Final Statistics:")
    print(f"Total Rows: {totalRows}")
    print(f"Total Tokens: {totalTokens}")
    print(f"GBs Written: {gbSpace:.2f}")
    print(f"Hours Elapsed: {elapsed / 3600:.2f}")