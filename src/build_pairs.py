"""Build training pairs or triplets from processed data for sentence-transformers fine-tuning.

    python -m src.build_pairs --config configs/pilot.yaml --split train --mode ingredients
    python -m src.build_pairs --config configs/pilot.yaml --split train --mode ingredients --negatives

Reads `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl` from `data/processed/<profile>/` and
formats pairs (query, positive) or triplets (query, positive, negative).
Negative recipes are sampled from the same split without sharing origin recipe IDs or groups.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import load_config, processed_dir
from src.contracts import QueryRecord, RecipeRecord
from src.embeddings import EmbeddingMode, MODES, format_document, format_query
from src.utils import REPO_ROOT, read_jsonl, write_jsonl


def load_corpus_dict(directory: Path | str) -> dict[str, RecipeRecord]:
    """Map recipe_id -> RecipeRecord from corpus.jsonl."""
    records = [RecipeRecord(**row) for row in read_jsonl(Path(directory) / "corpus.jsonl")]
    return {rec.recipe_id: rec for rec in records}


def load_qrels_dict(directory: Path | str) -> dict[str, dict[str, int]]:
    """Map query_id -> {recipe_id: relevance} from qrels.jsonl."""
    mapping: dict[str, dict[str, int]] = {}
    for row in read_jsonl(Path(directory) / "qrels.jsonl"):
        qid = row["query_id"]
        rid = row["recipe_id"]
        rel = int(row["relevance"])
        mapping.setdefault(qid, {})[rid] = rel
    return mapping


def build_pairs(
    directory: Path | str,
    *,
    split: str = "train",
    mode: EmbeddingMode = "ingredients",
    include_negatives: bool = False,
    hard_negatives: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Extract query-positive pairs (or triplets) for fine-tuning.

    Guarantees:
      - Only queries belonging to `split` are used.
      - Negatives never share origin_recipe_ids or origin_group_ids with the query.
      - Recipes are formatted according to `mode` ("ingredients" or "title_ingredients").
    """
    directory = Path(directory)
    corpus = load_corpus_dict(directory)
    qrels = load_qrels_dict(directory)
    queries = [
        QueryRecord(**row)
        for row in read_jsonl(directory / "queries.jsonl")
        if row.get("split") == split
    ]

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    # Recipes in this split available for negative candidate sampling
    candidate_recipe_ids = sorted(corpus.keys())

    for q in queries:
        q_positives = [
            rid
            for rid, rel in qrels.get(q.query_id, {}).items()
            if rel >= 1 and rid in corpus
        ]
        if not q_positives:
            continue

        q_text = format_query(q.ingredients)
        forbidden_ids = set(q.origin_recipe_ids)
        forbidden_groups = set(q.origin_group_ids)

        for pos_id in q_positives:
            pos_rec = corpus[pos_id]
            pos_text = format_document(pos_rec, mode)

            item: dict[str, Any] = {
                "query_id": q.query_id,
                "query": q_text,
                "recipe_id": pos_id,
                "positive": pos_text,
                "mode": mode,
            }

            if include_negatives:
                neg_id = _sample_negative(
                    q,
                    candidate_recipe_ids,
                    corpus,
                    forbidden_ids,
                    forbidden_groups,
                    hard_negatives=hard_negatives,
                    rng=rng,
                )
                if neg_id is not None:
                    item["negative_recipe_id"] = neg_id
                    item["negative"] = format_document(corpus[neg_id], mode)

            pairs.append(item)

    return pairs


def _sample_negative(
    query: QueryRecord,
    candidates: list[str],
    corpus: dict[str, RecipeRecord],
    forbidden_ids: set[str],
    forbidden_groups: set[str],
    *,
    hard_negatives: bool,
    rng: random.Random,
) -> str | None:
    """Sample a non-relevant recipe that does not share group or origin."""
    q_ingredients = set(query.ingredients)

    if hard_negatives:
        # Lexical hard negative: shares at least one ingredient but is forbidden neither by id nor group
        scored: list[tuple[int, str]] = []
        for rid in candidates:
            if rid in forbidden_ids:
                continue
            rec = corpus[rid]
            if rec.group_id in forbidden_groups:
                continue
            overlap = len(q_ingredients.intersection(rec.ingredients_normalized))
            if overlap > 0:
                scored.append((overlap, rid))

        if scored:
            scored.sort(key=lambda x: -x[0])
            top_k = scored[: min(10, len(scored))]
            return rng.choice(top_k)[1]

    # Random negative fallback
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    for rid in shuffled:
        if rid in forbidden_ids:
            continue
        rec = corpus[rid]
        if rec.group_id not in forbidden_groups:
            return rid
    return None


def save_pairs(pairs: list[dict[str, Any]], out_path: Path | str) -> None:
    """Write training pairs to a JSONL file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, pairs)


def load_pairs(in_path: Path | str) -> list[dict[str, Any]]:
    """Load training pairs from a JSONL file."""
    return list(read_jsonl(Path(in_path)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--mode", default="ingredients", choices=MODES)
    parser.add_argument("--negatives", action="store_true", help="Include sampled negative recipe in each pair")
    parser.add_argument("--hard-negatives", action="store_true", help="Mine lexical hard negatives")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: data/pairs/<split>_pairs.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg: DictConfig = load_config(args.config)
    directory = processed_dir(cfg)
    if not (directory / "corpus.jsonl").exists():
        print(f"error: {directory}/corpus.jsonl missing; build or fetch data first", file=sys.stderr)
        return 1

    pairs = build_pairs(
        directory,
        split=args.split,
        mode=args.mode,
        include_negatives=args.negatives or args.hard_negatives,
        hard_negatives=args.hard_negatives,
        seed=args.seed,
    )
    out_path = Path(args.out) if args.out else REPO_ROOT / "data" / "pairs" / f"{args.split}_{args.mode}_pairs.jsonl"
    save_pairs(pairs, out_path)
    print(f"Saved {len(pairs)} pairs to {out_path} (mode={args.mode}, negatives={args.negatives or args.hard_negatives})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
