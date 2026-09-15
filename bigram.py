import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import Callable


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, device: torch.device) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size, device=device)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, target=targets)
        return logits, loss

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
        splitter: Callable[[Tensor], tuple[Tensor, Tensor]],
        learning_rate: float = 1e-3,
        learning_steps: int = 10_000,
    ) -> None:
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for _ in range(learning_steps):
            xb, yb = splitter(dataset)
            _, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  # type: ignore

    @torch.no_grad()
    def estimate_loss(
        self,
        dataset: Tensor,
        splitter: Callable[[Tensor], tuple[Tensor, Tensor]],
        eval_steps: int = 100,
    ) -> float:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = splitter(dataset)
            _, loss = self(x, y)
            losses[k] = loss.item()
        return losses.mean().item()
