import torch

torch.manual_seed(1337) # type: ignore

INPUT_FILE_PATH = "data/input.txt"

BATCH_SIZE = 4
BLOCK_SIZE = 8


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor):
    data = train_data if split == 'train' else val_data

    # TODO: DEEP DIVE AND EXAMINE TO GET FULL UNDERSTANDING WHAT IS HAPPENING IN THE LINE BELLOW
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, )) 
    x = torch.stack([data[i: i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1: i+BLOCK_SIZE+1] for i in ix])
    return x, y


def main():
    print("="*20)
    print("LETS BUILD GPT FROM SCRATCH")
    print("="*20)

    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Text lenght: {len(text)}")
    print(f"Text sample: \n {text[:200]}")

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary: {chars}")
    print(f"Vocabulary size: {vocab_size}")

    stoi = { ch : i for i, ch in enumerate(chars)}
    itos = { i : ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s] # type: ignore
    decode = lambda l: [itos[i] for i in l] # type: ignore

    print("="*20)
    print("TOKENIZER PART")
    sample = "HELLO WORLD"
    print(f"ENCODE: {encode(sample)}")
    print(f"DECODE: {decode(encode(sample))}")
    print(f"ROUND TRIP VALID: {decode(encode(sample)) == list(sample)}")

    print("="*20)
    print("TENSOR PART")
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"SHAPE: {data.shape}, TYPE: {data.dtype}")
    print(f"SAMPLE: {data[:100]}")

    print("="*20)
    print("DATA SAMPLING")
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"TRAIN DATA SIZE: {len(train_data)}")
    print(f"VALIDATION DATA SIZE: {len(val_data)}")


    print("="*20)
    print("BLOCK SIZE")
    print(f"BATCH EXAMPLE: {train_data[:BLOCK_SIZE + 1]}")

    x = train_data[:BLOCK_SIZE]
    y = train_data[1:BLOCK_SIZE + 1]

    for t in range(BLOCK_SIZE):
        context = x[:t+1]
        target = y[t]
        print(f"CONTEXT: {context.tolist()}, EXPECTED PREDICTION: {target}") # type: ignore


        


        





if __name__ == "__main__":
    main()
