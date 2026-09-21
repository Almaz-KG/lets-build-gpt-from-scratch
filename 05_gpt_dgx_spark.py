import os

import torch
from torch import Tensor, nn
from torch.nn import functional as F

torch.manual_seed(1337)  # type: ignore


INPUT_FILE_PATH = "data/input.txt"

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_STEPS = 5_000
LEARNING_RATE = 3e-4

EVAL_STEPS = 100

BATCH_SIZE = 64
BLOCK_SIZE = 256

NUM_LAYERS = 6
NUM_HEADS = 6
NUM_EMBD = 384

DROPOUT = 0.2


class FeedForward(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p=DROPOUT),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Head(nn.Module):
    mask: torch.Tensor

    def __init__(self, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, head_size: int, n_embd: int, block_size: int
    ) -> None:
        super().__init__()

        assert num_heads * head_size == n_embd, (
            f"Embedding size {n_embd} must be equal to head size {head_size} multiplied by number of heads {num_heads}"
        )

        self.heads = nn.ModuleList(
            [
                Head(head_size=head_size, n_embd=n_embd, block_size=block_size)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
        )
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        n_embd: int,
        block_size: int,
        n_layer: int,
    ) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # (V, N)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd=n_embd, num_heads=num_heads, block_size=block_size)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=DEVICE)
        )  # (T, C)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits.view(B * T, logits.shape[-1]), target=targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, context: Tensor, new_tokens: int) -> Tensor:
        for _ in range(new_tokens):
            cond = context[:, -self.block_size :]
            logits, _ = self(cond)  # B, T, C
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return context

    def train_(
        self,
        dataset: Tensor,
        val_dataset: Tensor,
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
                tr = self.estimate_loss(dataset=dataset, eval_steps=EVAL_STEPS)
                va = self.estimate_loss(dataset=val_dataset, eval_steps=EVAL_STEPS)
                print(f"step: {i}, train: {tr:.4f}, val: {va:.4f}, gap: {va - tr:+.4f}")

    @torch.no_grad()
    def estimate_loss(
        self,
        dataset: Tensor,
        eval_steps: int = 100,
    ) -> float:
        is_training = self.training

        self.eval()
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = get_batch(dataset)
            _, loss = self(x, y)
            losses[k] = loss.item()

        self.train(is_training)
        return losses.mean().item()


def get_batch(data: Tensor) -> tuple[Tensor, Tensor]:
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

    print("=" * 20)
    print("MULTI HEAD LANGUAGE MODEL")
    m = MultiHeadModel(
        num_heads=NUM_HEADS,
        n_embd=NUM_EMBD,
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=NUM_LAYERS,
    ).to(torch.device(DEVICE))

    xb, yb = get_batch(train_data)

    b_logits, loss = m(xb, yb)
    print(f"LOGITS SHAPE: {b_logits.shape}")
    print(f"LOGITS LOSS: {loss}")
    print(f"EVAL LOSS: {m.estimate_loss(dataset=train_data, eval_steps=EVAL_STEPS)}")

    context = torch.zeros(
        (1, 1), dtype=torch.long, device=DEVICE
    )  # 0 = new line symbol

    b_gen = m.generate(context, new_tokens=20)[0].tolist()  # type: ignore
    print(f"GENERATION BEFORE TRAINING: {decode(b_gen)}")

    m.train_(dataset=train_data, val_dataset=val_data, learning_steps=LEARNING_STEPS)  # type: ignore
    b_gen2 = m.generate(context, new_tokens=100)[0].tolist()  # type: ignore
    print(f"GENERATION AFTER TRAINING: {decode(b_gen2)}")
    training_loss = m.estimate_loss(dataset=train_data)

    val_loss = m.estimate_loss(dataset=val_data)

    print(f"TRAINING LOSS: {training_loss:.2f}")
    print(f"VALIDATION LOSS: {val_loss:.2f}")
    torch.save(
        {
            "state_dict": m.state_dict(),
            "chars": chars,
            "config": {
                "n_embd": NUM_EMBD, "num_heads": NUM_HEADS, "n_layer": NUM_LAYERS,
                "block_size": BLOCK_SIZE, "dropout": DROPOUT, "vocab_size": vocab_size,
            },
            "losses": {"train": training_loss, "val": val_loss},
        },
        "checkpoints/05_gpt.pt",
    )


if __name__ == "__main__":
    main()
