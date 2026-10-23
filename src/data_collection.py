"""Build the processed datasets from the raw RecipeNLG CSV (raw file is only ever read).

    python -m src.data_collection --raw-path data/raw/RecipeNLG_dataset.csv --config configs/pilot.yaml

Pipeline: stream ids -> seeded shuffle sampling over the WHOLE file -> parse/validate/normalise ->
exact dedup -> group near-identical recipes -> group-wise 72/8/20 split -> synthetic queries.
`pilot` is the first n_pilot recipes of the main selection (nested subset, same splits).
Output per profile in data/processed/<profile>/: corpus.jsonl, splits.json, queries.jsonl,
qrels.jsonl, dataset_manifest.json.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit

from src.config import fingerprint, load_config, processed_dir
from src.config import raw_csv as resolve_raw_csv
from src.normalize import NORMALIZATION_FINGERPRINT, NORMALIZATION_VERSION
from src.preprocess import build_recipe, raw_int_id, title_key
from src.queries import generate_queries
from src.utils import sha256_file, sha256_obj, write_json, write_jsonl

MANIFEST_SCHEMA = "1"
ID_COLUMN = "Unnamed: 0"
COLUMNS = [ID_COLUMN, "title", "ingredients", "directions", "link", "source", "NER"]


def read_all_ids(raw_csv: Path, chunk_rows: int) -> np.ndarray:
    parts = [
        chunk[ID_COLUMN].to_numpy(dtype=np.int64)
        for chunk in pd.read_csv(raw_csv, usecols=[ID_COLUMN], chunksize=chunk_rows)
    ]
    return np.concatenate(parts)


def candidate_ids(all_ids: np.ndarray, seed: int, count: int) -> np.ndarray:
    """First `count` ids of a seeded shuffle of the whole file (= selection priority order).

    A larger `count` with the same seed extends the same permutation, so retries stay consistent.
    """
    return np.random.default_rng(seed).permutation(all_ids)[:count]


def load_rows(raw_csv: Path, wanted: np.ndarray, chunk_rows: int) -> dict[int, dict]:
    wanted_sorted = np.sort(wanted)
    rows: dict[int, dict] = {}
    reader = pd.read_csv(raw_csv, usecols=COLUMNS, chunksize=chunk_rows, dtype={ID_COLUMN: "int64"})
    for chunk in reader:
        mask = np.isin(chunk[ID_COLUMN].to_numpy(), wanted_sorted)
        for record in chunk[mask].to_dict("records"):
            rows[int(record[ID_COLUMN])] = record
    return rows


def select_recipes(
    rows: dict[int, dict], order: np.ndarray, n_target: int, data_cfg: dict
) -> tuple[list[dict], Counter, int]:
    """Walk candidates in key order; keep valid, non-duplicate recipes until n_target."""
    selected: list[dict] = []
    drops: Counter = Counter()
    seen: set[tuple] = set()
    examined = 0
    for raw_id in order:
        if len(selected) >= n_target:
            break
        examined += 1
        record, reason = build_recipe(
            int(raw_id),
            rows[int(raw_id)],
            min_ingredients=data_cfg["min_ingredients"],
            max_ingredients=data_cfg["max_ingredients"],
        )
        if record is None:
            drops[reason] += 1
            continue
        dup_key = (title_key(record["title"]), tuple(sorted(record["ingredients_normalized"])))
        if dup_key in seen:
            drops["exact_duplicate"] += 1
            continue
        seen.add(dup_key)
        selected.append(record)
    return selected, drops, examined


def assign_groups(records: list[dict]) -> dict:
    """Group recipes that share an identical ingredient set or an identical title key.

    Connected components of a record-key graph (scipy), so no all-pairs similarity: near-duplicates
    with a different title AND ingredient set are NOT caught (documented in the manifest).
    """
    key_node: dict[tuple, int] = {}
    rows, cols = [], []
    for index, record in enumerate(records):
        for key in (
            ("ing", tuple(sorted(record["ingredients_normalized"]))),
            ("title", title_key(record["title"])),
        ):
            rows.append(index)
            cols.append(len(records) + key_node.setdefault(key, len(key_node)))
    size = len(records) + len(key_node)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    labels = connected_components(graph, directed=False)[1][: len(records)]

    frame = pd.DataFrame({"label": labels, "raw_id": [raw_int_id(r["recipe_id"]) for r in records]})
    group_ids = "group:" + frame.groupby("label")["raw_id"].transform("min").astype(str)
    for record, group_id in zip(records, group_ids, strict=True):
        record["group_id"] = group_id
    sizes = frame["label"].value_counts()
    return {
        "n_groups": len(sizes),
        "max_group_size": int(sizes.iloc[0]),
        "mean_group_size": round(float(sizes.mean()), 3),
        "n_singleton_groups": int((sizes == 1).sum()),
        "top_group_sizes": sizes.head(10).tolist(),
    }


def split_by_group(
    records: list[dict], seed: int, test_fraction: float, validation_fraction: float
) -> dict[str, str]:
    """Whole groups go to one split (sklearn GroupShuffleSplit twice: test, then validation).

    Fractions apply to groups, so recipe-level shares are approximate (72/8/20).
    """
    groups = np.array([r["group_id"] for r in records])
    everything = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val, test = next(splitter.split(everything, groups=groups))
    validation_share = validation_fraction / (1 - test_fraction)
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_share, random_state=seed)
    train, validation = next(splitter.split(train_val, groups=groups[train_val]))
    split_of = {}
    for name, indices in (("test", test), ("validation", train_val[validation]), ("train", train_val[train])):
        split_of.update({records[i]["recipe_id"]: name for i in indices})
    return split_of


def _corpus_stats(records: list[dict]) -> dict:
    counts = [len(r["ingredients_normalized"]) for r in records]
    return {
        "ingredient_count": {
            "min": min(counts),
            "median": statistics.median(counts),
            "mean": round(statistics.fmean(counts), 3),
            "max": max(counts),
        },
        "sources": dict(Counter(r["source"] for r in records)),
        "empty_ner_fallback": sum(1 for r in records if not r["ner_raw"]),
    }


def write_profile(
    profile: str,
    records: list[dict],
    split_of: dict[str, str],
    cfg: DictConfig,
    out_dir: Path,
    shared_manifest: dict,
) -> dict:
    """Write one profile's files and return its manifest."""
    records = sorted(records, key=lambda r: raw_int_id(r["recipe_id"]))
    splits = {s: sorted(i for i in (r["recipe_id"] for r in records) if split_of[i] == s)
              for s in ("train", "validation", "test")}
    splits = {s: sorted(ids, key=raw_int_id) for s, ids in splits.items()}

    caps = {s: cfg["queries"]["caps"][s][profile] for s in ("train", "validation", "test")}
    queries, qrels, query_stats = generate_queries(
        records,
        split_of,
        seed=cfg.seed,
        min_recipe_ingredients=cfg["queries"]["min_recipe_ingredients"],
        min_query_size=cfg["queries"]["min_query_size"],
        max_query_size=cfg["queries"]["max_query_size"],
        caps=caps,
        common_df_threshold=cfg["queries"]["common_df_threshold"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "corpus.jsonl", records)
    write_json(
        out_dir / "splits.json",
        {"schema_version": MANIFEST_SCHEMA, "seed": cfg.seed, "profile": profile, **splits},
    )
    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "qrels.jsonl", qrels)

    manifest = {
        **shared_manifest,
        "profile": profile,
        "counts": {
            "recipes": len(records),
            **{f"recipes_{s}": len(ids) for s, ids in splits.items()},
            "groups": len({r["group_id"] for r in records}),
            "queries": len(queries),
            "qrels": len(qrels),
        },
        "queries": {"caps": caps, **query_stats},
        "stats": _corpus_stats(records),
        "files": {
            name: sha256_file(out_dir / name)
            for name in ("corpus.jsonl", "splits.json", "queries.jsonl", "qrels.jsonl")
        },
    }
    write_json(out_dir / "dataset_manifest.json", manifest)
    return manifest


def build(cfg: DictConfig, raw_csv: Path, profiles: list[str]) -> dict[str, dict]:
    data_cfg = cfg["data"]
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {raw_csv}\nPut RecipeNLG_dataset.csv in data/raw/ "
            f"(see README, 'Dataset') or pass --raw-path / set RECIPENLG_RAW."
        )
    chunk_rows = data_cfg["chunk_rows"]
    n_main = data_cfg["n_main"]

    all_ids = read_all_ids(raw_csv, chunk_rows)
    factor = float(data_cfg["candidate_factor"])
    while True:
        wanted = candidate_ids(all_ids, cfg.seed, math.ceil(n_main * factor))
        rows = load_rows(raw_csv, wanted, chunk_rows)
        records, drops, examined = select_recipes(rows, wanted, n_main, data_cfg)
        if len(records) >= n_main or len(wanted) >= len(all_ids):
            break
        factor *= 2

    group_info = assign_groups(records)
    split_of = split_by_group(records, cfg.seed, data_cfg["test_fraction"], data_cfg["validation_fraction"])

    shared = {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": cfg.seed,
        "config_sha256": fingerprint(cfg),
        "normalization": {"version": NORMALIZATION_VERSION, "fingerprint": NORMALIZATION_FINGERPRINT},
        "raw": {
            "file": raw_csv.name,
            "bytes": raw_csv.stat().st_size,
            "sha256": sha256_file(raw_csv),
            "rows": int(len(all_ids)),
        },
        "selection": {
            "target_main": n_main,
            "selected_main": len(records),
            "shortfall": max(0, n_main - len(records)),
            "candidate_factor_used": factor,
            "candidates_examined": examined,
            "drop_reasons": dict(drops),
            "sampling": "seeded permutation of all raw ids over the whole file; not the first rows",
        },
        "groups": {
            **group_info,
            "rule": "union of identical normalised-ingredient-set and identical title key",
            "limitation": "no all-pairs similarity; near-duplicates with different title and "
            "ingredient set stay in different groups",
        },
        "split": {
            "method": "sklearn GroupShuffleSplit on groups: test, then validation from the rest",
            "test_fraction": data_cfg["test_fraction"],
            "validation_fraction": data_cfg["validation_fraction"],
        },
        "selection_fingerprint": sha256_obj([r["recipe_id"] for r in records]),
    }

    sizes = {"main": len(records), "pilot": min(data_cfg["n_pilot"], len(records))}
    manifests: dict[str, dict] = {}
    for profile in profiles:
        subset = records[: sizes[profile]]  # records are still in key (selection) order here
        manifests[profile] = write_profile(
            profile, subset, split_of, cfg, processed_dir(cfg, profile), shared
        )
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw-path", help="RecipeNLG_dataset.csv (default: config / $RECIPENLG_RAW)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profiles", nargs="+", default=["pilot", "main"], choices=["pilot", "main"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    manifests = build(cfg, resolve_raw_csv(cfg, args.raw_path), args.profiles)
    for profile, manifest in manifests.items():
        print(f"[{profile}] {manifest['counts']}")
    first = next(iter(manifests.values()))
    print("drop reasons:", first["selection"]["drop_reasons"])
    print("groups:", {k: first["groups"][k] for k in ("n_groups", "max_group_size", "top_group_sizes")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
