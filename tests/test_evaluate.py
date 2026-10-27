import mlflow
import pytest

from src.baseline import fit_retriever, load_bundle, save_bundle
from src.config import load_config, processed_dir
from src.contracts import SearchHit
from src.evaluate import evaluate_retriever, load_split, log_to_mlflow, main
from src.utils import read_json, read_jsonl


class FakeRetriever:
    name, version = "fake", "1"

    def __init__(self, hits_by_ingredient):
        self.hits = hits_by_ingredient

    def search(self, ingredients, top_k):
        return [SearchHit(rid, 1.0) for rid in self.hits[ingredients[0]]][:top_k]


def test_scores_match_hand_computation():
    # q1: right answer at rank 1; q2: at rank 2; q3: no results at all.
    retriever = FakeRetriever({"a": ["r1", "r5"], "b": ["r7", "r2"], "c": []})
    queries = [
        {"query_id": "q1", "ingredients": ["a"]},
        {"query_id": "q2", "ingredients": ["b"]},
        {"query_id": "q3", "ingredients": ["c"]},
    ]
    qrels = {"q1": {"r1": 1}, "q2": {"r2": 1}, "q3": {"r9": 1}}
    result = evaluate_retriever(retriever, queries, qrels)
    assert result["recall@5"] == pytest.approx(2 / 3)
    assert result["recall@10"] == pytest.approx(2 / 3)
    assert result["mrr@10"] == pytest.approx((1 + 0.5 + 0) / 3)  # empty result counts as 0, not skipped
    assert result["n_queries"] == 3


def test_retriever_order_is_kept_even_when_scores_tie():
    # both hits have score 1.0; the retriever put r2 first, so MRR must be 1, not decided by ranx ties
    retriever = FakeRetriever({"a": ["r2", "r1"]})
    result = evaluate_retriever(retriever, [{"query_id": "q", "ingredients": ["a"]}], {"q": {"r2": 1}})
    assert result["mrr@10"] == 1.0


def test_load_split_only_returns_that_split(built_fixture):
    cfg, _, _ = built_fixture
    directory = processed_dir(cfg, "main")
    queries, qrels = load_split(directory, "validation")
    assert queries and all(q["split"] == "validation" for q in queries)
    assert set(qrels) == {q["query_id"] for q in queries}


def test_end_to_end_numbers_equal_mlflow(built_fixture, tmp_path):
    cfg, _, _ = built_fixture
    directory = processed_dir(cfg, "main")
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    train_ids = set(read_json(directory / "splits.json")["train"])
    retriever, info = fit_retriever("tfidf", corpus, train_ids)
    bundle = save_bundle(retriever, info, cfg, tmp_path)
    loaded, manifest = load_bundle(bundle)

    queries, qrels = load_split(directory, "train")
    results = evaluate_retriever(loaded, queries, qrels)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_id = log_to_mlflow(results, manifest, bundle, cfg, "train", 10, uri, "test-exp")

    logged = mlflow.MlflowClient(uri).get_run(run_id).data
    for name in ("recall@5", "recall@10", "mrr@10"):
        assert logged.metrics[name.replace("@", "_at_")] == pytest.approx(results[name])
    assert logged.metrics["fit_seconds"] == pytest.approx(info["fit_seconds"])
    assert logged.params["model"] == "tfidf" and logged.params["split"] == "train"


def test_test_split_needs_explicit_flag():
    with pytest.raises(SystemExit):
        main(["--bundle", "does-not-matter", "--split", "test"])


@pytest.mark.slow
def test_cli_on_real_pilot(tmp_path, capsys):
    cfg = load_config("configs/pilot.yaml")
    directory = processed_dir(cfg, "pilot")
    if not (directory / "corpus.jsonl").exists():
        pytest.skip("pilot data not built")
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    retriever, info = fit_retriever("tfidf", corpus, set(read_json(directory / "splits.json")["train"]))
    bundle = save_bundle(retriever, info, cfg, tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    assert main(["--bundle", str(bundle), "--split", "validation", "--tracking-uri", uri]) == 0
    printed = capsys.readouterr().out
    assert "recall@10" in printed and "mlflow run:" in printed
