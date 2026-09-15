import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor

from bigram import BigramLanguageModel

torch.manual_seed(1337)  # type: ignore

INPUT_FILE_PATH = "data/input.txt"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # I DON'T HAVE CUDA
# DEVICE = 'mps' if torch.mps.is_available() else 'cpu'
DEVICE = "cpu"

LEARNING_STEPS = 10_000
EVAL_STEPS = 10_000

BATCH_SIZE = 4
BLOCK_SIZE = 8

VOCAB_SIZE = 65


def get_batch(split: str, train_data: Tensor | None, val_data: Tensor | None):
    data = train_data if split == "train" else val_data

    assert data is not None

    # TODO: DEEP DIVE AND EXAMINE TO GET FULL UNDERSTANDING WHAT IS HAPPENING IN THE LINE BELLOW
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
    return x, y


def main():
    print("=" * 20)
    print("LETS BUILD GPT FROM SCRATCH")
    print(f"DEVICE: {DEVICE}")
    print("=" * 20)

    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text lenght: {len(text)}")
    print(f"Text sample: \n {text[:200]}")

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary: {chars}")
    print(f"Vocabulary size: {vocab_size}")
    assert vocab_size == VOCAB_SIZE

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]  # type: ignore
    decode = lambda l: [itos[i] for i in l]  # type: ignore

    print("=" * 20)
    print("TOKENIZER PART")
    sample = "HELLO WORLD"
    print(f"ENCODE: {encode(sample)}")
    print(f"DECODE: {decode(encode(sample))}")
    print(f"ROUND TRIP VALID: {decode(encode(sample)) == list(sample)}")

    print("=" * 20)
    print("TENSOR PART")
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"SHAPE: {data.shape}, TYPE: {data.dtype}")
    print(f"SAMPLE: {data[:100]}")

    print("=" * 20)
    print("DATA SAMPLING")
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"TRAIN DATA SIZE: {len(train_data)}")
    print(f"VALIDATION DATA SIZE: {len(val_data)}")

    print("=" * 20)
    print("BLOCK SIZE")
    print(f"BATCH EXAMPLE: {train_data[: BLOCK_SIZE + 1]}")

    x = train_data[:BLOCK_SIZE]
    y = train_data[1 : BLOCK_SIZE + 1]

    for t in range(BLOCK_SIZE):
        context = x[: t + 1]
        target = y[t]
        print(f"CONTEXT: {context.tolist()}, EXPECTED PREDICTION: {target}")  # type: ignore

    print("=" * 20)
    print("BIGRAM LANGUAGE MODEL")
    m = BigramLanguageModel(vocab_size=VOCAB_SIZE, device=torch.device(DEVICE))

    xb, yb = get_batch("train", train_data=train_data, val_data=val_data)

    b_logits, _ = m(xb, yb)
    print(f"LOGITS SHAPE: {b_logits.shape}")
    splitter = lambda x: get_batch("train", x, None)  # type: ignore
    print(
        f"LOGITS LOSS: {m.estimate_loss(dataset=train_data, splitter=splitter, eval_steps=EVAL_STEPS)}"
    )  # type: ignore

    context = torch.zeros(
        (1, 1), dtype=torch.long, device=DEVICE
    )  # 0 = new line symbol

    # splitter = lambda x: get_batch("train", x, None) # type: ignore
    b_gen = m.generate(context, new_tokens=20)[0].tolist()  # type: ignore
    print(f"BIGRAM GENERATION BEFORE TRAINING: {''.join(decode(b_gen))}")

    m.train_(dataset=train_data, splitter=splitter)  # type: ignore
    b_gen2 = m.generate(context, new_tokens=100)[0].tolist()  # type: ignore
    print(f"BIGRAM GENERATION AFTER TRAINING: {''.join(decode(b_gen2))}")
    training_loss = m.estimate_loss(dataset=train_data, splitter=splitter)  # type: ignore

    val_splitter = lambda x: get_batch("val", train_data=None, val_data=x)  # type: ignore
    val_loss = m.estimate_loss(dataset=val_data, splitter=val_splitter)  # type: ignore

    print(f"TRAINING LOSS: {training_loss:.2f}")  # type: ignore
    print(f"VALIDATION LOSS: {val_loss:.2f}")  # type: ignore


if __name__ == "__main__":
    main()
