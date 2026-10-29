from collections import Counter

import pandas as pd
import pytest

from src.baseline import fit_retriever, save_bundle
from src.config import processed_dir
from src.contracts import SearchHit
from src.evaluate import (
    ANNOTATORS,
    MANUAL_DIR,
    POOL_COLUMNS,
    assign_annotators,
    build_pool,
    load_system,
    main,
)
from src.utils import read_json, read_jsonl, write_json


class ListRetriever:
    name, version = "list", "1"

    def __init__(self, ranked):
        self.ranked = ranked

    def search(self, ingredients, top_k):
        return [SearchHit(rid, 1.0) for rid in self.ranked][:top_k]


def recipe(rid):
    return {"title": f"t{rid}", "ingredients_raw": ["1 egg"], "directions": ["Mix."], "link": "https://x"}


QUERY = {"query_id": "m01", "text": "eggs please", "ingredients": "egg;milk"}
CORPUS = {rid: recipe(rid) for rid in ("r1", "r2", "r3", "r4")}


def test_pool_merges_duplicates_and_hides_systems():
    systems = {"SYS_A": ListRetriever(["r1", "r2", "r3"]), "SYS_B": ListRetriever(["r3", "r4"])}
    items, key = build_pool(systems, [QUERY], CORPUS, depth=3)
    assert sorted(i["recipe_id"] for i in items) == ["r1", "r2", "r3", "r4"]  # r3 pooled once
    assert all(list(item) == POOL_COLUMNS for item in items)
    assert not any("SYS_" in str(value) for item in items for value in item.values())
    assert {k["recipe_id"]: k["systems"] for k in key}["r3"] == "SYS_A@3 SYS_B@1"
    assert [i["item_id"] for i in items] == [k["item_id"] for k in key]


def test_pool_depth_and_seed():
    systems = {"A": ListRetriever(["r1", "r2", "r3", "r4"])}
    first, _ = build_pool(systems, [QUERY], CORPUS, depth=2, seed=7)
    again, _ = build_pool(systems, [QUERY], CORPUS, depth=2, seed=7)
    assert len(first) == 2 and first == again


def test_pool_columns_match_the_label_template():
    assert POOL_COLUMNS == list(pd.read_csv(MANUAL_DIR / "labels_template.csv").columns)


def test_annotators_round_robin_and_double_labels():
    items = [{"query_id": f"m{q:02d}", "item_id": f"m{q:02d}-{n}"} for q in range(1, 21) for n in range(12)]
    rows = assign_annotators(items)
    owners = {}
    for row in rows:
        owners.setdefault(row["query_id"], row["annotator"])
    assert Counter(owners.values()) == {"hoang": 7, "hung": 7, "hieu": 6}
    per_item = Counter(row["item_id"] for row in rows)
    for query_id in owners:
        doubled = [i for i, n in per_item.items() if i.startswith(query_id) and n == 2]
        assert len(doubled) >= 0.2 * 12
    for item_id, n in per_item.items():
        labellers = [r["annotator"] for r in rows if r["item_id"] == item_id]
        assert len(set(labellers)) == n  # a double label always comes from a second person


def test_fine_tuned_bundle_needs_its_encoder(tmp_path):
    write_json(tmp_path / "manifest.json", {"model_name": "minilm_finetuned"})
    with pytest.raises(ValueError, match="checkpoint"):
        load_system(tmp_path)


@pytest.fixture(scope="module")
def fixture_bundles(built_fixture, tmp_path_factory):
    cfg, _, out = built_fixture
    directory = processed_dir(cfg, "main")
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    train_ids = set(read_json(directory / "splits.json")["train"])
    tmp = tmp_path_factory.mktemp("pool")
    config = tmp / "fixture.yaml"
    config.write_text(
        f"paths:\n  processed_dir: {out.as_posix()}\ndata:\n  profile: main\n", encoding="utf-8"
    )
    from omegaconf import OmegaConf

    main_cfg = OmegaConf.merge(cfg, {"data": {"profile": "main"}})
    bundles = {
        m: save_bundle(*fit_retriever(m, corpus, train_ids), main_cfg, tmp / "models")
        for m in ("tfidf", "bm25")
    }
    return config, bundles, tmp


def test_load_system_reports_what_fit_time_measures(fixture_bundles):
    _, bundles, _ = fixture_bundles
    _, manifest, seconds, what = load_system(bundles["bm25"])
    assert manifest.model_name == "bm25" and seconds >= 0 and what == "index build"


def test_pool_cli_writes_one_blind_file_per_annotator(fixture_bundles, capsys):
    config, bundles, tmp = fixture_bundles
    args = ["pool", "--config", str(config), "--out", str(tmp / "out")]
    args += [arg for label, path in bundles.items() for arg in ("--system", f"{label.upper()}={path}")]
    assert main(args) == 0
    files = {a: pd.read_csv(tmp / "out" / f"pool_dev_{a}.csv", dtype=str) for a in ANNOTATORS}
    key = pd.read_csv(tmp / "out" / "pool_dev_key.csv")
    labelled = set().union(*(set(f["item_id"]) for f in files.values()))
    assert labelled == set(key["item_id"])  # every pooled item has an annotator
    assert all(list(f.columns) == POOL_COLUMNS and (f["annotator"] == a).all() for a, f in files.items())
    assert "pooled items for 20 queries" in capsys.readouterr().out


def test_evaluate_cli_without_mlflow(fixture_bundles, capsys):
    config, bundles, _ = fixture_bundles
    argv = [
        "--bundle",
        str(bundles["tfidf"]),
        "--split",
        "validation",
        "--config",
        str(config),
        "--no-mlflow",
    ]
    assert main(argv) == 0
    assert "mrr@10" in capsys.readouterr().out


def test_pool_cli_preserves_existing_labels(fixture_bundles):
    config, bundles, tmp = fixture_bundles
    out = tmp / "labelled"
    out.mkdir()
    labelled = out / "pool_dev_hoang.csv"
    original = "item_id,relevance,notes\nm01-01,2,checked\n"
    labelled.write_text(original, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["pool", "--config", str(config), "--system", f"E01={bundles['tfidf']}", "--out", str(out)])
    assert labelled.read_text(encoding="utf-8") == original
