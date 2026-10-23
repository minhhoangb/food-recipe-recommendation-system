"""Check that data/processed/<profile> is complete, matches its manifest and has no split leakage.

    python scripts/verify_data.py            # both profiles that exist
    python scripts/verify_data.py pilot      # one profile
Exit code 1 on any problem, so it can gate CI/Colab cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, processed_dir  # noqa: E402
from src.contracts import QrelRecord, QueryRecord, RecipeRecord  # noqa: E402
from src.utils import read_json, read_jsonl, sha256_file  # noqa: E402

SPLITS = ("train", "validation", "test")


def verify_profile(directory: Path) -> list[str]:
    problems: list[str] = []
    manifest_path = directory / "dataset_manifest.json"
    if not manifest_path.exists():
        return [f"{directory}: dataset_manifest.json missing (run src.data_collection or fetch)"]
    manifest = read_json(manifest_path)
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256_file(path) != expected:
            problems.append(f"{name}: sha256 differs from manifest")
    if problems:
        return problems

    corpus = [RecipeRecord(**r) for r in read_jsonl(directory / "corpus.jsonl")]
    by_id = {r.recipe_id: r for r in corpus}
    if len(by_id) != len(corpus):
        problems.append("corpus: duplicate recipe_id")
    splits = read_json(directory / "splits.json")
    split_of: dict[str, str] = {}
    for split in SPLITS:
        for recipe_id in splits[split]:
            if recipe_id in split_of:
                problems.append(f"splits: {recipe_id} appears in two splits")
            split_of[recipe_id] = split
    if set(split_of) != set(by_id):
        problems.append("splits: ids differ from corpus ids")
    group_split: dict[str, str] = {}
    for recipe in corpus:
        previous = group_split.setdefault(recipe.group_id, split_of[recipe.recipe_id])
        if previous != split_of[recipe.recipe_id]:
            problems.append(f"group {recipe.group_id} spans two splits")
            break

    queries = {q.query_id: q for q in (QueryRecord(**r) for r in read_jsonl(directory / "queries.jsonl"))}
    for query in queries.values():
        if any(split_of.get(o) != query.split for o in query.origin_recipe_ids):
            problems.append(f"query {query.query_id}: origin not in its split")
            break
    for qrel in (QrelRecord(**r) for r in read_jsonl(directory / "qrels.jsonl")):
        if qrel.query_id not in queries or qrel.recipe_id not in by_id:
            problems.append(f"qrel {qrel.query_id}/{qrel.recipe_id}: dangling reference")
            break
    return problems


def main(argv: list[str]) -> int:
    cfg = load_config()
    profiles = argv or [p for p in ("pilot", "main") if processed_dir(cfg, p).exists()]
    if not profiles:
        print("No processed data found. Run: python -m src.data_collection  (or scripts/fetch_artifacts.py)")
        return 1
    failed = False
    for profile in profiles:
        problems = verify_profile(processed_dir(cfg, profile))
        print(f"[{profile}] {'OK' if not problems else 'PROBLEMS'}")
        for problem in problems:
            print("   -", problem)
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
