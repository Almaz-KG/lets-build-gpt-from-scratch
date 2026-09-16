import torch
from torch import Tensor, nn
from torch.nn import functional as F

torch.manual_seed(1337)  # type: ignore


INPUT_FILE_PATH = "data/input.txt"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # I DON'T HAVE CUDA
# DEVICE = 'mps' if torch.mps.is_available() else 'cpu'
DEVICE = "cpu"

LEARNING_STEPS = 10_000
LEARNING_RATE = 1e-3

EVAL_STEPS = 100

BATCH_SIZE = 4
BLOCK_SIZE = 8

# VOCAB_SIZE = 65


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.token_embedding_table(idx)

        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits.view(B * T, C), target=targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, context: Tensor, new_tokens: int) -> Tensor:
        for _ in range(new_tokens):
            logits, _ = self(context)  # B, T, C
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return context

    def train_(
        self,
        dataset: Tensor,
        learning_rate: float = LEARNING_RATE,
        learning_steps: int = LEARNING_STEPS,
    ) -> None:
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for i in range(learning_steps):
            xb, yb = get_batch(dataset)
            _, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  # type: ignore

            if i % 500 == 0:
                el = self.estimate_loss(dataset=dataset, eval_steps=EVAL_STEPS)
                print(f"step: {i}, loss: {el:.4f}")

    @torch.no_grad()
    def estimate_loss(
        self,
        dataset: Tensor,
        eval_steps: int = 100,
    ) -> float:
        self.eval()
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = get_batch(dataset)
            _, loss = self(x, y)
            losses[k] = loss.item()

        self.train()
        return losses.mean().item()


def get_batch(data: Tensor) -> tuple[Tensor, Tensor]:
    # TODO: DEEP DIVE AND EXAMINE TO GET FULL UNDERSTANDING WHAT IS HAPPENING IN THE LINE BELOW
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

    print(f"Text length: {len(text)}")
    print(f"Text sample: \n {text[:200]}")

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary: {chars}")
    print(f"Vocabulary size: {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]  # type: ignore
    decode = lambda l: "".join([itos[i] for i in l])  # type: ignore

    print("=" * 20)
    print("TOKENIZER PART")
    sample = "HELLO WORLD"
    print(f"ENCODE: {encode(sample)}")
    print(f"DECODE: {decode(encode(sample))}")
    print(f"ROUND TRIP VALID: {decode(encode(sample)) == sample}")

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
    m = BigramLanguageModel(vocab_size=vocab_size).to(torch.device(DEVICE))

    xb, yb = get_batch(train_data)

    b_logits, _ = m(xb, yb)
    print(f"LOGITS SHAPE: {b_logits.shape}")
    print(f"LOGITS LOSS: {m.estimate_loss(dataset=train_data, eval_steps=EVAL_STEPS)}")

    context = torch.zeros(
        (1, 1), dtype=torch.long, device=DEVICE
    )  # 0 = new line symbol

    b_gen = m.generate(context, new_tokens=20)[0].tolist()  # type: ignore
    print(f"BIGRAM GENERATION BEFORE TRAINING: {decode(b_gen)}")

    m.train_(dataset=train_data, learning_steps=LEARNING_STEPS)  # type: ignore
    b_gen2 = m.generate(context, new_tokens=100)[0].tolist()  # type: ignore
    print(f"BIGRAM GENERATION AFTER TRAINING: {decode(b_gen2)}")
    training_loss = m.estimate_loss(dataset=train_data)

    val_loss = m.estimate_loss(dataset=val_data)

    print(f"TRAINING LOSS: {training_loss:.2f}")
    print(f"VALIDATION LOSS: {val_loss:.2f}")


if __name__ == "__main__":
    main()
