"""Build the blind pool of the manual evaluation.

    python scripts/build_pool.py --split dev --out reports/manual_eval/pool_dev.csv

For every query of the split, the top 5 recipes of every frozen system are taken, merged,
deduplicated and shuffled. The pool never names the system that produced a recipe: which
systems are behind a row is written to a separate file (`pool_dev_systems.csv`), so the
annotator cannot recognise them and the file is only opened when the per-system coverage
is reported.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving import load_retriever  # noqa: E402
from src.utils import REPO_ROOT, read_jsonl, write_jsonl  # noqa: E402

POOL_TOP_K = 5
COLUMNS = (
    "query_id",
    "query_text",
    "query_ingredients",
    "item_id",
    "recipe_id",
    "title",
    "ingredients",
    "directions",
    "link",
    "relevance",
    "annotator",
    "notes",
)


def read_queries(path: Path, split: str) -> list[dict]:
    """The queries of one split, read from the CSV the annotation guide describes."""
    import csv

    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["split"] == split]


def recipe_rows(corpus_path: Path) -> dict[str, dict]:
    """recipe_id -> the recipe, for the pool rows."""
    return {row["recipe_id"]: row for row in read_jsonl(corpus_path)}


def build_pool(
    queries: list[dict],
    systems: dict[str, Path],
    corpus: dict[str, dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """(pool rows for the annotator, the systems behind every row)."""
    retrievers = {name: load_retriever(bundle) for name, bundle in systems.items()}
    rng = random.Random(seed)
    pool: list[dict] = []
    origin: list[dict] = []
    for query in queries:
        query_id = query["query_id"]
        found: dict[str, list[str]] = {}
        for name, retriever in retrievers.items():
            hits = retriever.search(query["ingredients"].split(";"), POOL_TOP_K)
            found[name] = [hit.recipe_id for hit in hits]
        recipe_ids = list({recipe_id for ids in found.values() for recipe_id in ids})
        rng.shuffle(recipe_ids)
        for index, recipe_id in enumerate(recipe_ids, start=1):
            recipe = corpus.get(recipe_id, {})
            pool.append(
                {
                    "query_id": query_id,
                    "query_text": query["text"],
                    "query_ingredients": query["ingredients"],
                    "item_id": f"{query_id}-i{index:02d}",
                    "recipe_id": recipe_id,
                    "title": recipe.get("title", ""),
                    "ingredients": "; ".join(recipe.get("ingredients_normalized", [])),
                    "directions": " | ".join(recipe.get("directions", [])),
                    "link": recipe.get("link", ""),
                    "relevance": "",
                    "annotator": "",
                    "notes": "",
                }
            )
            origin.append(
                {
                    "query_id": query_id,
                    "item_id": f"{query_id}-i{index:02d}",
                    "recipe_id": recipe_id,
                    "systems": ",".join(name for name, ids in found.items() if recipe_id in ids),
                }
            )
    return pool, origin


def write_csv(path: Path, rows: list[dict]) -> int:
    """The pool as a CSV with the columns of `labels_template.csv` (the annotator fills them)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--queries", default="reports/manual_eval/queries.csv")
    parser.add_argument("--corpus", default="data/processed/main/corpus.jsonl")
    parser.add_argument("--split", default="dev", choices=["dev", "final"])
    parser.add_argument("--system", action="append", default=[], help="NAME=models/<run_id>")
    parser.add_argument("--out-dir", default="reports/manual_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    queries = read_queries(REPO_ROOT / args.queries, args.split)
    if not queries:
        print(f"no {args.split} queries in {args.queries}")
        return 1
    systems = {}
    for spec in args.system:
        name, _, path = spec.partition("=")
        systems[name] = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    if not systems:
        print("give at least one --system NAME=models/<run_id>")
        return 1

    corpus = recipe_rows(REPO_ROOT / args.corpus)
    pool, origin = build_pool(queries, systems, corpus, args.seed)
    out_dir = REPO_ROOT / args.out_dir
    write_csv(out_dir / f"pool_{args.split}.csv", pool)
    write_jsonl(out_dir / f"pool_{args.split}_systems.jsonl", origin)
    print(f"{len(pool)} rows over {len(queries)} queries -> {out_dir / f'pool_{args.split}.csv'}")
    print(f"systems behind the rows -> {out_dir / f'pool_{args.split}_systems.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
