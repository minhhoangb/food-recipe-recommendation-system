"""Synthetic query + qrels generation (origin-recipe retrieval proxy).

A query is a random 2-6 subset of a recipe's normalised ingredients, with at least one ingredient
hidden. The only known positive is the origin recipe, so metrics built on these labels are a
*proxy*: other recipes that also fit the query count as misses. Queries never cross splits.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.preprocess import raw_int_id

SPLIT_SALT = {"train": 11, "validation": 22, "test": 33}


def _recipe_order_key(raw_id: int, seed: int, split: str) -> int:
    rng = np.random.default_rng([seed, raw_id, SPLIT_SALT[split], 0])
    return int(rng.integers(0, 2**62))


def make_query_ingredients(
    ingredients: list[str],
    raw_id: int,
    split: str,
    seed: int,
    min_size: int,
    max_size: int,
    specific: set[str] | None = None,
) -> list[str]:
    """Random 2-6 subset hiding >=1 ingredient; includes one `specific` (rare) ingredient if given."""
    n = len(ingredients)
    upper = min(max_size, n - 1)  # always hide at least one ingredient
    rng = np.random.default_rng([seed, raw_id, SPLIT_SALT[split], 1])
    size = int(rng.integers(min_size, upper + 1))
    anchors = [i for i, name in enumerate(ingredients) if specific and name in specific]
    if anchors:
        anchor = int(rng.choice(anchors))
        others = [i for i in range(n) if i != anchor]
        rest = rng.choice(others, size=size - 1, replace=False).tolist()
        chosen = sorted([anchor, *rest])
    else:
        chosen = sorted(rng.choice(n, size=size, replace=False).tolist())
    return [ingredients[i] for i in chosen]


def generate_queries(
    records: list[dict],
    split_of: dict[str, str],
    *,
    seed: int,
    min_recipe_ingredients: int,
    min_query_size: int,
    max_query_size: int,
    caps: dict[str, int],
    common_df_threshold: float = 0.02,
) -> tuple[list[dict], list[dict], dict]:
    """Return (queries, qrels, stats). Deterministic for a fixed corpus/split/config.

    Ingredients appearing in more than `common_df_threshold` of the corpus (salt, sugar, ...) are
    "common"; every query must contain at least one non-common ingredient, otherwise "salt, water"
    style queries would have thousands of equally valid answers and make the metrics meaningless.
    """
    stats: dict = {
        "skipped_short_recipes": 0,
        "skipped_only_common_ingredients": 0,
        "merged_same_split": 0,
        "dropped_cross_split": 0,
    }
    doc_freq: dict[str, int] = defaultdict(int)
    for record in records:
        for name in set(record["ingredients_normalized"]):
            doc_freq[name] += 1
    limit = common_df_threshold * len(records)
    specific_of = {
        r["recipe_id"]: {n for n in r["ingredients_normalized"] if doc_freq[n] <= limit}
        for r in records
    }
    stats["common_df_threshold"] = common_df_threshold

    by_split: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        split = split_of[record["recipe_id"]]
        if len(record["ingredients_normalized"]) < max(min_recipe_ingredients, min_query_size + 1):
            stats["skipped_short_recipes"] += 1
            continue
        if not specific_of[record["recipe_id"]]:
            stats["skipped_only_common_ingredients"] += 1
            continue
        by_split[split].append(record)

    candidates: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    order_of: dict[str, int] = {}
    for split, split_records in by_split.items():
        cap = caps.get(split, 0)
        ordered = sorted(
            split_records,
            key=lambda r: _recipe_order_key(raw_int_id(r["recipe_id"]), seed, split),
        )
        for rank, record in enumerate(ordered[: int(cap * 1.5) + 10]):
            raw_id = raw_int_id(record["recipe_id"])
            ingredients = make_query_ingredients(
                record["ingredients_normalized"],
                raw_id,
                split,
                seed,
                min_query_size,
                max_query_size,
                specific_of[record["recipe_id"]],
            )
            key = tuple(sorted(ingredients))
            query = {
                "query_id": f"q:{raw_id}",
                "split": split,
                "ingredients": ingredients,
                "origin_recipe_ids": [record["recipe_id"]],
                "origin_group_ids": [record["group_id"]],
                "label_source": "synthetic",
            }
            candidates[key][split].append(query)
            order_of[query["query_id"]] = rank

    queries_by_split: dict[str, list[dict]] = defaultdict(list)
    for per_split in candidates.values():
        if len(per_split) > 1:  # same ingredient set generated in different splits
            stats["dropped_cross_split"] += sum(len(v) for v in per_split.values())
            continue
        (split, same_key), = per_split.items()
        first = same_key[0]
        for extra in same_key[1:]:
            stats["merged_same_split"] += 1
            first["origin_recipe_ids"] += extra["origin_recipe_ids"]
            first["origin_group_ids"] += extra["origin_group_ids"]
        first["origin_group_ids"] = sorted(set(first["origin_group_ids"]))
        queries_by_split[split].append(first)

    queries: list[dict] = []
    for split in ("train", "validation", "test"):
        kept = sorted(queries_by_split[split], key=lambda q: order_of[q["query_id"]])
        queries.extend(kept[: caps.get(split, 0)])
    queries.sort(key=lambda q: (q["split"], q["query_id"]))

    qrels = [
        {
            "query_id": q["query_id"],
            "recipe_id": recipe_id,
            "relevance": 1,
            "label_source": "synthetic_origin",
        }
        for q in queries
        for recipe_id in sorted(q["origin_recipe_ids"])
    ]
    stats["queries_per_split"] = {s: sum(q["split"] == s for q in queries) for s in SPLIT_SALT}
    return queries, qrels, stats
