import math
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src import final_evaluation as final
from src.baseline import fit_retriever, save_bundle
from src.evaluate import POOL_COLUMNS, assign_annotators, build_pool, write_csv
from src.manual_evaluation import evaluate_labels, read_csv
from src.utils import read_json, read_jsonl, write_json
from tests.test_pool import ListRetriever


@pytest.fixture
def frozen_setup(built_fixture, tmp_path):
    cfg, _, directory = built_fixture
    corpus = list(read_jsonl(directory / "main/corpus.jsonl"))
    ids = set(read_json(directory / "main/splits.json")["train"])
    systems = {}
    for label, name in (("E01", "tfidf"), ("E03", "bm25")):
        retriever, info = fit_retriever(name, corpus, ids)
        systems[label] = str(save_bundle(retriever, info, cfg, tmp_path / "models"))
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    expected = sum(q["split"] == "test" for q in read_jsonl(directory / "main/queries.jsonl"))
    out = tmp_path / "final"
    final.freeze(str(config), systems, {}, "E03", out, expected_queries=expected)
    return out / "frozen_config.json", tmp_path, expected


def test_e10_runs_once_and_scores_saved_rankings(frozen_setup, monkeypatch):
    path, tmp, expected = frozen_setup
    calls = []
    original = final.run_retriever

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(final, "run_retriever", counted)
    result = final.run_final(path, f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    assert len(calls) == 2
    assert set(result.test_n_queries) == {expected}
    assert result.set_index("experiment").loc["E03", "selected_deployment"]
    assert read_json(path.parent / "e10_execution.json")["status"] == "complete"
    per_query = pd.read_csv(path.parent / "comparison_test_per_query.csv")
    means = per_query.groupby("experiment")["mrr@10"].mean()
    for row in result.to_dict("records"):
        assert row["test_mrr@10"] == pytest.approx(means[row["experiment"]])
    with pytest.raises(FileExistsError, match="already started"):
        final.run_final(path, f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    assert len(calls) == 2


def test_modified_bundle_is_rejected_before_test_read(frozen_setup, monkeypatch):
    path, tmp, _ = frozen_setup
    bundle = Path(read_json(path)["systems"][0]["bundle"])
    with (bundle / "model.joblib").open("ab") as handle:
        handle.write(b"changed")
    monkeypatch.setattr(final, "load_split", lambda *args: pytest.fail("test must not be read"))
    with pytest.raises(ValueError, match="bundle changed"):
        final.run_final(path, f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    assert not (path.parent / "e10_execution.json").exists()


def test_failed_e10_remains_blocked(frozen_setup, monkeypatch):
    path, tmp, _ = frozen_setup
    monkeypatch.setattr(
        final, "run_retriever", lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted"))
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        final.run_final(path, f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    assert read_json(path.parent / "e10_execution.json")["status"] == "failed_no_automatic_retry"
    with pytest.raises(FileExistsError):
        final.run_final(path, f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")


def make_label_files(tmp, complete=True):
    corpus = {
        r: {"title": r, "ingredients_raw": ["egg"], "directions": ["Cook"], "link": "https://x"}
        for r in ("r1", "r2")
    }
    queries = [{"query_id": f"q{i}", "text": "egg", "ingredients": "egg"} for i in range(3)]
    items, _ = build_pool({"A": ListRetriever(["r1", "r2"])}, queries, corpus)
    assignments = assign_annotators(items)
    labels = tmp / "labels"
    labels.mkdir()
    write_csv(labels / "pool_final.csv", items, POOL_COLUMNS)
    counts = {}
    for name in ("hoang", "hung", "hieu"):
        rows = [r for r in assignments if r["annotator"] == name]
        if complete:
            for row in rows:
                row["relevance"] = "2" if row["recipe_id"] == "r1" else "0"
        write_csv(labels / f"pool_final_{name}.csv", rows, POOL_COLUMNS)
        counts[name] = len(rows)
    write_json(labels / "pool_final_metadata.json", {"assignments": counts})
    logs = [
        {
            "annotator": name,
            "label_source": "human",
            "independent": "yes",
            "exposure": "none",
            "completed_utc": "fixture only",
        }
        for name in counts
    ]
    write_csv(labels / "annotation_log.csv", logs, list(logs[0]))
    keys = tmp / "keys"
    keys.mkdir()
    write_json(
        keys / "system_runs_top10.json",
        {"A": {q["query_id"]: {"unjudged": 3.0, "r1": 2.0, "r2": 1.0} for q in queries}},
    )
    return labels, keys, tmp / "results"


def test_manual_metrics_wait_for_labels_without_reading_keys(tmp_path):
    labels, keys, out = make_label_files(tmp_path, complete=False)
    status = evaluate_labels(labels, keys / "missing", out)
    assert not status["metrics_published"]
    assert sum(status["missing_labels"].values()) > 0


def test_pooled_ndcg_reports_unjudged_coverage_and_preserves_originals(tmp_path):
    labels, keys, out = make_label_files(tmp_path)
    status = evaluate_labels(labels, keys, out)
    assert status["metrics_published"]
    assert status["scores"][0]["mean_top10_judging_coverage"] == pytest.approx(2 / 3)
    assert status["scores"][0]["pooled_ndcg_at_10"] == pytest.approx(1 / math.log2(3))
    with (labels / "pool_final_hoang.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="changed after locking"):
        evaluate_labels(labels, keys, out)


def test_disagreements_require_adjudication_without_changing_originals(tmp_path):
    labels, keys, out = make_label_files(tmp_path)
    assignments = {n: read_csv(labels / f"pool_final_{n}.csv") for n in ("hoang", "hung", "hieu")}
    item_id = next(
        r["item_id"]
        for r in assignments["hoang"]
        if any(x["item_id"] == r["item_id"] for x in assignments["hung"])
    )
    changed = next(r for r in assignments["hoang"] if r["item_id"] == item_id)
    changed["relevance"] = "1"
    write_csv(labels / "pool_final_hoang.csv", assignments["hoang"], POOL_COLUMNS)
    status = evaluate_labels(labels, keys, out)
    assert status["status"] == "awaiting_adjudication" and not status["metrics_published"]
    adjudications = read_csv(labels / "adjudication.csv")
    for row in adjudications:
        row.update(relevance="2", adjudicator="fixture human", reason="fixture joint discussion")
    write_csv(labels / "adjudication.csv", adjudications, list(adjudications[0]))
    assert evaluate_labels(labels, keys, out)["metrics_published"]
    # Erasing a locked original must fail before the incomplete-label gate.
    changed["relevance"] = ""
    write_csv(labels / "pool_final_hoang.csv", assignments["hoang"], POOL_COLUMNS)
    with pytest.raises(ValueError, match="changed after locking"):
        evaluate_labels(labels, keys, out)


def test_exposed_labels_cannot_be_counted_as_independent(tmp_path):
    labels, keys, out = make_label_files(tmp_path)
    provenance = read_csv(labels / "annotation_log.csv")
    provenance[0]["exposure"] = "saw AI suggestions"
    write_csv(labels / "annotation_log.csv", provenance, list(provenance[0]))
    with pytest.raises(ValueError, match="provenance is incomplete"):
        evaluate_labels(labels, keys / "not-readable", out)
