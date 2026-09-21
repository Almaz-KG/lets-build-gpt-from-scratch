"""Why eight identical heads drift apart, and what would make them collapse.

Two configurations of the same stage-03 model, both with all eight heads
initialised bitwise identically:

  cat  the heads are concatenated, so head i owns channels [4i : 4i+4]
  sum  the heads are added, so every head writes into the same channels

Each is measured twice: the pairwise cosine between head gradients after a
single backward, before any optimiser step, and the divergence between the
head weights after a full training run.

The model code below is a copy of 03_multi_head.py, kept standalone the same
way the stages are.
"""

import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_FILE_PATH = "data/input.txt"

DEVICE = "cpu"

SEED = 1337

LEARNING_STEPS = 10_000
LEARNING_RATE = 1e-3

EVAL_STEPS = 100

BATCH_SIZE = 4
BLOCK_SIZE = 8

NUM_HEADS = 8
NUM_EMBD = 32


def get_batch(dataset: Tensor) -> tuple[Tensor, Tensor]:
    ix = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([dataset[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([dataset[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x, y


class FeedForward(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Head(nn.Module):
    mask: torch.Tensor

    def __init__(self, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.register_buffer("mask", torch.tril(torch.ones((block_size, block_size))))
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    """The stage-03 wiring: head i owns its own slice of the output vector."""

    def __init__(
        self, num_heads: int, n_embd: int, head_size: int, block_size: int
    ) -> None:
        super().__init__()
        assert num_heads * head_size == n_embd, (
            f"Embedding size {n_embd} must be equal to head size {head_size} multiplied by number of heads {num_heads}"
        )

        self.heads = nn.ModuleList(
            Head(head_size=head_size, n_embd=n_embd, block_size=block_size)
            for _ in range(num_heads)
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([h(x) for h in self.heads], dim=-1)


class SumHeadAttention(nn.Module):
    """The collapsing wiring: every head writes into every output channel.

    Adding the outputs instead of concatenating them forces head_size == n_embd,
    because the sum keeps the width of a single head instead of stacking them.
    """

    def __init__(
        self, num_heads: int, n_embd: int, head_size: int, block_size: int
    ) -> None:
        super().__init__()
        assert head_size == n_embd, (
            f"Summed heads each write the full vector, so head size {head_size} must equal embedding size {n_embd}"
        )

        self.heads = nn.ModuleList(
            Head(head_size=head_size, n_embd=n_embd, block_size=block_size)
            for _ in range(num_heads)
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([h(x) for h in self.heads], dim=0).sum(dim=0)


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        vocab_size: int,
        block_size: int,
        attention_cls: type[nn.Module] = MultiHeadAttention,
    ) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = attention_cls(
            num_heads=num_heads,
            n_embd=n_embd,
            head_size=head_size,
            block_size=block_size,
        )
        self.ffwd = FeedForward(n_embd=n_embd)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))

        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        x = self.ffwd(x)

        logits = self.lm_head(x)
        if targets is not None:
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits.view(B * T, logits.shape[-1]), target=targets)
            return logits, loss

        return logits, None

    def train_(
        self,
        dataset: Tensor,
        learning_rate: float = LEARNING_RATE,
        learning_steps: int = LEARNING_STEPS,
        eval_steps: int = EVAL_STEPS,
    ):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for i in range(learning_steps):
            xb, yb = get_batch(dataset)
            _, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  # type: ignore

            if i % 2000 == 0:
                el = self.estimate_loss(dataset=dataset, eval_steps=eval_steps)
                print(f"    step {i:>5}, loss {el:.4f}")

    @torch.no_grad
    def estimate_loss(self, dataset: Tensor, eval_steps: int = EVAL_STEPS) -> float:
        is_training = self.training

        self.eval()
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = get_batch(dataset)
            _, loss = self(x, y)
            losses[k] = loss.item()

        self.train(is_training)
        return losses.mean().item()


def get_file_content(file_name: str) -> str:
    with open(file_name, "r", encoding="utf-8") as f:
        return f.read()


def tie_heads(attention: nn.Module) -> None:
    """Overwrite every head with head 0, so the eight are bitwise identical."""
    source = attention.heads[0].state_dict()  # type: ignore
    for head in attention.heads[1:]:  # type: ignore
        head.load_state_dict(source)  # type: ignore


def heads_are_identical(attention: nn.Module) -> bool:
    reference = attention.heads[0].state_dict()  # type: ignore
    return all(
        torch.equal(reference[name], tensor)  # type: ignore
        for head in attention.heads[1:]  # type: ignore
        for name, tensor in head.state_dict().items()  # type: ignore
    )


def max_head_divergence(attention: nn.Module) -> float:
    """The largest absolute difference between head 0 and any other head."""
    reference = attention.heads[0].state_dict()  # type: ignore
    return max(
        (tensor - reference[name]).abs().max().item()  # type: ignore
        for head in attention.heads[1:]  # type: ignore
        for name, tensor in head.state_dict().items()  # type: ignore
    )


def head_gradients(attention: nn.Module) -> Tensor:
    """One flat gradient vector per head: key, query and value concatenated."""
    return torch.stack(
        [
            torch.cat(
                [
                    head.key.weight.grad.flatten(),  # type: ignore
                    head.query.weight.grad.flatten(),  # type: ignore
                    head.value.weight.grad.flatten(),  # type: ignore
                ]
            )
            for head in attention.heads  # type: ignore
        ]
    )


def pairwise_cosines(vectors: Tensor) -> Tensor:
    normalised = F.normalize(vectors, dim=-1)
    similarity = normalised @ normalised.T
    rows, cols = torch.triu_indices(len(vectors), len(vectors), offset=1)
    return similarity[rows, cols]


def build_model(
    attention_cls: type[nn.Module], head_size: int, vocab_size: int
) -> MultiHeadModel:
    """Same seed for both configurations, then the heads are tied."""
    torch.manual_seed(SEED)  # type: ignore
    model = MultiHeadModel(
        num_heads=NUM_HEADS,
        head_size=head_size,
        n_embd=NUM_EMBD,
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        attention_cls=attention_cls,
    ).to(torch.device(DEVICE))
    tie_heads(model.sa_heads)
    return model


CONFIGURATIONS = (
    ("cat", MultiHeadAttention, NUM_EMBD // NUM_HEADS),
    ("sum", SumHeadAttention, NUM_EMBD),
)


def main():
    text = get_file_content(INPUT_FILE_PATH)

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(text))
    train_data = data[:n]
    val_data = data[n:]

    torch.manual_seed(SEED)  # type: ignore
    xb, yb = get_batch(train_data)

    print("=" * 72)
    print("ONE BACKWARD, NO OPTIMISER STEP")
    print("=" * 72)

    for label, attention_cls, head_size in CONFIGURATIONS:
        model = build_model(attention_cls, head_size, vocab_size)
        _, loss = model(xb, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore

        cosines = pairwise_cosines(head_gradients(model.sa_heads))
        parameters = sum(p.numel() for p in model.parameters())

        print(f"\n[{label}] head_size {head_size}, {parameters} parameters")
        print(
            f"  heads identical before backward: {heads_are_identical(model.sa_heads)}"
        )
        print(f"  loss: {loss.item():.4f}")  # type: ignore
        print(
            f"  pairwise cosine over {len(cosines)} pairs: "
            f"mean {cosines.mean():+.4f}, min {cosines.min():+.4f}, max {cosines.max():+.4f}"
        )

    print()
    print("=" * 72)
    print(f"AFTER {LEARNING_STEPS} TRAINING STEPS")
    print("=" * 72)

    for label, attention_cls, head_size in CONFIGURATIONS:
        print(f"\n[{label}] training")
        model = build_model(attention_cls, head_size, vocab_size)

        torch.manual_seed(SEED)  # type: ignore
        started = time.time()
        model.train_(dataset=train_data, learning_steps=LEARNING_STEPS)
        elapsed = time.time() - started

        train_loss = model.estimate_loss(dataset=train_data)
        val_loss = model.estimate_loss(dataset=val_data)

        print(f"  train {train_loss:.4f}, val {val_loss:.4f}, {elapsed:.0f} s")
        print(f"  heads still identical: {heads_are_identical(model.sa_heads)}")
        print(f"  max head divergence:   {max_head_divergence(model.sa_heads):.2e}")


if __name__ == "__main__":
    main()
