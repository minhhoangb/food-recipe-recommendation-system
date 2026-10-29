"""Build training pairs from train recipes with make_query_ingredients and Hưng's document formatting.

    python -m src.build_pairs --config configs/pilot.yaml --split train \
        --mode ingredients --max-pairs 10000 --seed 42

Guarantees:
  1. Only recipes from the specified split (train) are used; never validation or test recipes.
  2. Query ingredients are sampled via src.queries.make_query_ingredients using multiple seeds.
  3. Positive recipes are formatted using src.embeddings.format_document (E04/E05).
  4. Saves pairs.jsonl and a manifest with sha256 hash.
  5. Provides NoDuplicateBatchSampler ensuring no batch contains duplicate recipes or duplicate groups.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from omegaconf import DictConfig
from torch.utils.data import Sampler

from src.config import load_config, processed_dir
from src.contracts import RecipeRecord
from src.embeddings import MODES, EmbeddingMode, format_document, format_query
from src.preprocess import raw_int_id
from src.queries import make_query_ingredients
from src.utils import REPO_ROOT, read_json, read_jsonl, sha256_file, write_json, write_jsonl

DEFAULT_PAIRS_DIR = REPO_ROOT / "data" / "pairs"


def build_pairs(
    directory: Path | str,
    *,
    split: str = "train",
    mode: EmbeddingMode = "ingredients",
    max_pairs: int = 10000,
    seed: int = 42,
    min_recipe_ingredients: int = 3,
    min_query_size: int = 2,
    max_query_size: int = 6,
    common_df_threshold: float = 0.02,
    queries_per_recipe: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build training pairs from the specified split (default: train).

    Never reads recipes from validation or test splits.
    Returns (pairs, stats).
    """
    directory = Path(directory)
    splits = read_json(directory / "splits.json")
    allowed_ids = set(splits.get(split, []))

    all_records = [RecipeRecord(**row) for row in read_jsonl(directory / "corpus.jsonl")]

    # Calculate document frequency over corpus to identify specific (rare) ingredients
    doc_freq: dict[str, int] = defaultdict(int)
    for r in all_records:
        for ing in set(r.ingredients_normalized):
            doc_freq[ing] += 1
    df_limit = max(1.0, common_df_threshold * len(all_records))
    specific_of = {
        r.recipe_id: {ing for ing in r.ingredients_normalized if doc_freq[ing] <= df_limit}
        or set(r.ingredients_normalized)
        for r in all_records
    }

    # Filter to requested split only
    split_recipes = [r for r in all_records if r.recipe_id in allowed_ids]

    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Iterate over recipes and sample queries with multiple seed offsets
    rng = random.Random(seed)
    shuffled_recipes = split_recipes[:]
    rng.shuffle(shuffled_recipes)

    for offset in range(queries_per_recipe):
        current_seed = seed + offset * 7919
        for r in shuffled_recipes:
            if len(pairs) >= max_pairs:
                break
            if len(r.ingredients_normalized) < max(min_recipe_ingredients, min_query_size + 1):
                continue
            if not specific_of.get(r.recipe_id):
                continue

            raw_id = raw_int_id(r.recipe_id)
            q_ingredients = make_query_ingredients(
                r.ingredients_normalized,
                raw_id,
                split,
                current_seed,
                min_query_size,
                max_query_size,
                specific=specific_of[r.recipe_id],
            )
            q_text = format_query(q_ingredients)
            pos_text = format_document(r, mode)

            dedup_key = (r.recipe_id, q_text)
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)

            pairs.append(
                {
                    "pair_id": f"pair-{len(pairs) + 1:06d}",
                    "query": q_text,
                    "positive": pos_text,
                    "recipe_id": r.recipe_id,
                    "group_id": r.group_id,
                    "split": split,
                    "mode": mode,
                    "seed": current_seed,
                }
            )

        if len(pairs) >= max_pairs:
            break

    stats = {
        "split": split,
        "mode": mode,
        "total_pairs": len(pairs),
        "unique_recipes": len({p["recipe_id"] for p in pairs}),
        "unique_groups": len({p["group_id"] for p in pairs}),
        "seed": seed,
        "max_pairs": max_pairs,
    }
    return pairs, stats


def save_pairs_and_manifest(
    pairs: list[dict[str, Any]],
    stats: dict[str, Any],
    directory: Path | str,
    *,
    out_dir: Path | str = DEFAULT_PAIRS_DIR,
    profile: str = "pilot",
) -> tuple[Path, Path]:
    """Write pairs.jsonl and pairs_manifest.json with verified SHA256 hashes."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mode = stats["mode"]
    split = stats["split"]
    filename_stem = f"{profile}_{split}_{mode}_pairs"
    pairs_file = out_path / f"{filename_stem}.jsonl"
    manifest_file = out_path / f"{filename_stem}_manifest.json"

    write_jsonl(pairs_file, pairs)
    pairs_sha256 = sha256_file(pairs_file)

    corpus_file = Path(directory) / "corpus.jsonl"
    splits_file = Path(directory) / "splits.json"

    manifest = {
        "schema_version": "1",
        "profile": profile,
        "split": split,
        "mode": mode,
        "n_pairs": len(pairs),
        "pairs_file": pairs_file.name,
        "pairs_sha256": pairs_sha256,
        "corpus_sha256": sha256_file(corpus_file) if corpus_file.exists() else "",
        "splits_sha256": sha256_file(splits_file) if splits_file.exists() else "",
        "stats": stats,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(manifest_file, manifest)
    return pairs_file, manifest_file


def load_pairs(in_path: Path | str) -> list[dict[str, Any]]:
    """Load pairs from a JSONL file."""
    return list(read_jsonl(Path(in_path)))


# ---------------------------------------------------------------------------------------------
# Batch Sampler: No duplicate recipe or group in any batch
# ---------------------------------------------------------------------------------------------
class NoDuplicateBatchSampler(Sampler[list[int]]):
    """Yields batches such that no batch contains duplicate recipe_ids or group_ids.

    Essential for MultipleNegativesRankingLoss where in-batch negatives must not be
    positive for or belong to the same recipe group as another query in the batch.
    """

    def __init__(
        self,
        pairs: list[dict[str, Any]],
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.pairs = pairs
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        indices = list(range(len(self.pairs)))
        rng.shuffle(indices)

        batches: list[list[int]] = []
        current_batch: list[int] = []
        seen_recipes: set[str] = set()
        seen_groups: set[str] = set()
        leftover: list[int] = []

        for idx in indices:
            pair = self.pairs[idx]
            r_id = pair["recipe_id"]
            g_id = pair.get("group_id", r_id)

            if r_id in seen_recipes or (g_id and g_id in seen_groups):
                leftover.append(idx)
                continue

            current_batch.append(idx)
            seen_recipes.add(r_id)
            if g_id:
                seen_groups.add(g_id)

            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
                seen_recipes.clear()
                seen_groups.clear()

        # Handle leftovers by packaging into compatible batches
        while leftover:
            batch: list[int] = []
            seen_r: set[str] = set()
            seen_g: set[str] = set()
            unplaced: list[int] = []

            for idx in leftover:
                pair = self.pairs[idx]
                r_id = pair["recipe_id"]
                g_id = pair.get("group_id", r_id)

                if r_id in seen_r or (g_id and g_id in seen_g):
                    unplaced.append(idx)
                    continue

                batch.append(idx)
                seen_r.add(r_id)
                if g_id:
                    seen_g.add(g_id)

                if len(batch) == self.batch_size:
                    break

            if len(batch) == self.batch_size or (not self.drop_last and batch):
                batches.append(batch)
            if len(unplaced) == len(leftover):
                # Cannot place remaining items without duplicates
                if not self.drop_last and unplaced:
                    batches.append(unplaced[: self.batch_size])
                break
            leftover = [i for i in leftover if i not in batch]

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        for b in batches:
            yield b

    def __len__(self) -> int:
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--mode", default="ingredients", choices=list(MODES))
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(DEFAULT_PAIRS_DIR))
    args = parser.parse_args(argv)

    cfg: DictConfig = load_config(args.config)
    directory = processed_dir(cfg)
    if not (directory / "corpus.jsonl").exists():
        print(f"error: {directory}/corpus.jsonl missing; build or fetch data first", file=sys.stderr)
        return 1

    pairs, stats = build_pairs(
        directory,
        split=args.split,
        mode=args.mode,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    pairs_file, manifest_file = save_pairs_and_manifest(
        pairs,
        stats,
        directory,
        out_dir=args.out_dir,
        profile=str(cfg.data.profile),
    )
    print(f"Saved {len(pairs)} pairs to {pairs_file}")
    print(f"Manifest written to {manifest_file}")
    print(f"Stats: {json.dumps(stats, indent=2)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
