"""Fine-tune the MiniLM sentence-transformers encoder on recipe pairs or triplets.

    python -m src.finetune --config configs/pilot.yaml --epochs 1 --batch-size 16

Trains with MultipleNegativesRankingLoss on pairs (query, positive) or triplets
(query, positive, negative). Runs on CPU or CUDA (Colab GPU).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
try:
    from sentence_transformers.sentence_transformer import losses
except (ImportError, AttributeError):
    from sentence_transformers import losses
from sentence_transformers import InputExample, SentenceTransformer
from torch.utils.data import DataLoader

from src.build_pairs import build_pairs, load_pairs
from src.config import load_config, processed_dir
from src.embeddings import MODEL_NAME
from src.memory import peak_rss_mb
from src.utils import REPO_ROOT

DEFAULT_OUTPUT_MODEL = REPO_ROOT / "models" / "minilm_finetuned_checkpoint"


def prepare_examples(pairs: list[dict[str, Any]]) -> list[InputExample]:
    """Convert raw pair/triplet dicts to sentence-transformers InputExample objects."""
    examples: list[InputExample] = []
    for p in pairs:
        query = p["query"]
        pos = p["positive"]
        if "negative" in p and p["negative"]:
            examples.append(InputExample(texts=[query, pos, p["negative"]]))
        else:
            examples.append(InputExample(texts=[query, pos]))
    return examples


def finetune_model(
    pairs: list[dict[str, Any]],
    *,
    base_model_name: str = MODEL_NAME,
    output_path: Path | str | None = None,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    device: str | None = None,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    """Fine-tune a SentenceTransformer model on the provided pairs.

    Returns the fine-tuned model and a dictionary with training metadata.
    """
    if not pairs:
        raise ValueError("pairs list must not be empty for fine-tuning")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(base_model_name, device=device)
    examples = prepare_examples(pairs)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    start_time = time.perf_counter()
    start_rss = peak_rss_mb()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        show_progress_bar=False,
    )

    train_seconds = round(time.perf_counter() - start_time, 3)
    final_rss = peak_rss_mb()

    out_dir = Path(output_path) if output_path else DEFAULT_OUTPUT_MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))

    summary = {
        "base_model": base_model_name,
        "n_examples": len(examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "device": device,
        "train_seconds": train_seconds,
        "peak_rss_mb": round(final_rss, 1),
        "output_path": str(out_dir),
    }
    return model, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml")
    parser.add_argument("--pairs", default=None, help="Path to training pairs JSONL (if already built)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out-model", default=None)
    args = parser.parse_args(argv)

    cfg: DictConfig = load_config(args.config)
    if args.pairs:
        pairs = load_pairs(args.pairs)
    else:
        directory = processed_dir(cfg)
        if not (directory / "corpus.jsonl").exists():
            print(f"error: {directory}/corpus.jsonl missing; build or fetch data first", file=sys.stderr)
            return 1
        pairs = build_pairs(directory, split="train", mode="ingredients")

    print(f"Fine-tuning {MODEL_NAME} on {len(pairs)} pairs (epochs={args.epochs}, batch_size={args.batch_size})...")
    _, summary = finetune_model(
        pairs,
        output_path=args.out_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print("Fine-tuning completed successfully:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
