from collections import defaultdict

import numpy as np

from src.config import processed_dir
from src.data_collection import build, candidate_ids
from src.utils import read_json, read_jsonl
from tests.conftest import small_config


def _load(cfg, profile):
    d = processed_dir(cfg, profile)
    corpus = list(read_jsonl(d / "corpus.jsonl"))
    return {
        "corpus": corpus,
        "splits": read_json(d / "splits.json"),
        "queries": list(read_jsonl(d / "queries.jsonl")),
        "qrels": list(read_jsonl(d / "qrels.jsonl")),
        "manifest": read_json(d / "dataset_manifest.json"),
    }


def test_all_valid_recipes_selected_and_bad_rows_dropped(tmp_path, fixture_csv):
    cfg = small_config(tmp_path, n_main=1000, n_pilot=10)
    manifests = build(cfg, fixture_csv, ["main"])
    selection = manifests["main"]["selection"]
    assert selection["selected_main"] == 30
    assert selection["shortfall"] == 970  # real number is reported, not the target
    assert selection["drop_reasons"] == {
        "malformed_ingredients": 2,  # broken text + the eval-style string (never executed)
        "missing_field": 2,  # empty title and empty link become NaN
        "empty_directions": 1,
        "ingredient_count_out_of_range": 1,
        "exact_duplicate": 1,
    }


def test_pilot_is_nested_subset_with_same_splits(built_fixture):
    cfg, manifests, _ = built_fixture
    pilot, main = _load(cfg, "pilot"), _load(cfg, "main")
    assert manifests["main"]["counts"]["recipes"] == 25
    assert manifests["pilot"]["counts"]["recipes"] == 10
    main_split = {i: s for s in ("train", "validation", "test") for i in main["splits"][s]}
    for s in ("train", "validation", "test"):
        for recipe_id in pilot["splits"][s]:
            assert main_split[recipe_id] == s


def test_groups_never_cross_splits_and_ids_are_stable(tmp_path, fixture_csv):
    cfg = small_config(tmp_path, n_main=1000)  # keep all 30 so the duplicate group is complete
    build(cfg, fixture_csv, ["main"])
    main = _load(cfg, "main")
    split_of = {i: s for s in ("train", "validation", "test") for i in main["splits"][s]}
    group_splits, group_sizes = defaultdict(set), defaultdict(int)
    for record in main["corpus"]:
        group_splits[record["group_id"]].add(split_of[record["recipe_id"]])
        group_sizes[record["group_id"]] += 1
        assert record["recipe_id"].startswith("recipenlg:")
        assert record["link"].startswith("https://")
    assert max(group_sizes.values()) >= 2, "fixture must contain a multi-recipe group"
    assert all(len(v) == 1 for v in group_splits.values())


def test_identical_ingredient_sets_share_a_group(tmp_path, fixture_csv):
    cfg = small_config(tmp_path, n_main=1000)
    build(cfg, fixture_csv, ["main"])
    by_id = {r["recipe_id"]: r for r in _load(cfg, "main")["corpus"]}
    # "Exact Duplicate" (id 28) has the same ingredient set as recipe 0; id 29 is an exact duplicate of
    # recipe 0, so exactly one of 0/29 survives dedup (which one depends on the seeded order).
    survivor = by_id.get("recipenlg:0") or by_id["recipenlg:29"]
    assert by_id["recipenlg:28"]["group_id"] == survivor["group_id"]


def test_queries_respect_splits_and_hide_an_ingredient(built_fixture):
    cfg, _, _ = built_fixture
    main = _load(cfg, "main")
    by_id = {r["recipe_id"]: r for r in main["corpus"]}
    split_of = {i: s for s in ("train", "validation", "test") for i in main["splits"][s]}
    keys = defaultdict(set)
    assert main["queries"], "fixture should produce queries"
    for query in main["queries"]:
        assert 2 <= len(query["ingredients"]) <= 6
        for origin in query["origin_recipe_ids"]:
            assert split_of[origin] == query["split"]
            full = by_id[origin]["ingredients_normalized"]
            assert set(query["ingredients"]) < set(full)  # strict subset: something is hidden
        keys[tuple(sorted(query["ingredients"]))].add(query["split"])
    assert all(len(v) == 1 for v in keys.values()), "same query text in two splits"
    query_ids = {q["query_id"] for q in main["queries"]}
    assert {r["query_id"] for r in main["qrels"]} == query_ids
    assert all(r["relevance"] == 1 for r in main["qrels"])


def test_build_is_reproducible(tmp_path, fixture_csv):
    first = build(small_config(tmp_path / "a"), fixture_csv, ["pilot", "main"])
    second = build(small_config(tmp_path / "b"), fixture_csv, ["pilot", "main"])
    for profile in ("pilot", "main"):
        assert first[profile]["files"] == second[profile]["files"]
        assert first[profile]["selection_fingerprint"] == second[profile]["selection_fingerprint"]


def test_different_seed_changes_selection(tmp_path, fixture_csv):
    base = build(small_config(tmp_path / "a"), fixture_csv, ["main"])
    other_cfg = small_config(tmp_path / "b")
    other_cfg.seed = 7
    other = build(other_cfg, fixture_csv, ["main"])
    assert base["main"]["selection_fingerprint"] != other["main"]["selection_fingerprint"]


def test_manifest_has_provenance(built_fixture):
    _, manifests, _ = built_fixture
    manifest = manifests["main"]
    assert len(manifest["raw"]["sha256"]) == 64 and manifest["raw"]["rows"] == 37
    assert len(manifest["normalization"]["fingerprint"]) == 64
    assert set(manifest["files"]) == {"corpus.jsonl", "splits.json", "queries.jsonl", "qrels.jsonl"}
    assert "limitation" in manifest["groups"]


def test_sampling_spreads_over_whole_range_and_extends_consistently():
    ids = np.arange(0, 2_000_000, dtype=np.int64)
    chosen = candidate_ids(ids, 42, 1000)
    assert chosen.max() > 1_500_000 and chosen.min() < 500_000  # not "the first rows"
    assert (candidate_ids(ids, 42, 2000)[:1000] == chosen).all()  # bigger retry pool is a superset
