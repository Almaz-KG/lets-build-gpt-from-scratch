"""Sample from a trained checkpoint.

Not a stage: the model lives in the stage file and is imported from it, so the
weights are always read back into the exact architecture that produced them.
"""

import argparse
import importlib.util
import os
import sys
from types import ModuleType

import torch
from torch import Tensor

DEFAULT_CHECKPOINT_PATH = "checkpoints/05_gpt.pt"
DEFAULT_STAGE_PATH = "05_gpt_dgx_spark.py"


def resolve_device() -> str:
    if device := os.environ.get("DEVICE"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_stage(stage_path: str, device: str) -> ModuleType:
    """Import a stage file by path.

    A plain import will not do: the stage file names start with a digit. The
    stage reads its own DEVICE constant inside forward(), so it has to be set
    before the module body runs, or position indices land on the wrong device.
    """
    os.environ["DEVICE"] = device

    spec = importlib.util.spec_from_file_location("stage", stage_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import a stage from {stage_path}")

    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)  # main() is guarded, so nothing trains here
    return stage


def build_model(stage: ModuleType, checkpoint: dict, device: str) -> torch.nn.Module:
    config = checkpoint["config"]

    # dropout is a module-level constant in the stage rather than a constructor
    # argument, so the saved value cannot be passed in. eval() below makes that
    # moot for sampling.
    model = stage.MultiHeadModel(
        vocab_size=config["vocab_size"],
        num_heads=config["num_heads"],
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()  # without this the trained dropout stays on and degrades samples
    return model


def encode_prompt(prompt: str, stoi: dict[str, int], device: str) -> Tensor:
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        raise SystemExit(
            f"Prompt uses characters outside the trained vocabulary: {unknown}"
        )

    tokens = [stoi[c] for c in prompt]
    return torch.tensor([tokens], dtype=torch.long, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--stage", default=DEFAULT_STAGE_PATH)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument(
        "--prompt",
        default="",
        help="Priming text; defaults to a single newline, as during training",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device()
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,  # the weights were trained and saved on cuda
        weights_only=True,
    )

    stage = load_stage(args.stage, device)
    model = build_model(stage, checkpoint, device)

    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda tokens: "".join([itos[i] for i in tokens])

    losses = checkpoint["losses"]
    print(f"DEVICE: {device}", file=sys.stderr)
    print(f"CHECKPOINT: {args.checkpoint}", file=sys.stderr)
    print(
        f"TRAINED TO: train {losses['train']:.4f}, val {losses['val']:.4f}",
        file=sys.stderr,
    )
    print(
        f"PARAMETERS: {sum(p.numel() for p in model.parameters()):,}",
        file=sys.stderr,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    context = encode_prompt(args.prompt or "\n", stoi, device)
    generated = model.generate(context, new_tokens=args.tokens)[0].tolist()
    print(decode(generated))


if __name__ == "__main__":
    main()
