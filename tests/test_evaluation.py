"""Tests for the E04/E05 evaluation driver (metrics, comparison table, split loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation import METRICS, compare, evaluate_retriever, load_split
from src.utils import write_jsonl


class StubRetriever:
    """Returns the recipes whose id is in the query's ingredient list, in a fixed order."""

    name = "stub"
    version = "1"

    def __init__(self, ranking: dict[str, list[str]]):
        self.ranking = ranking

    def search(self, ingredients, top_k):
        from src.contracts import SearchHit

        key = " ".join(ingredients)
        ids = self.ranking.get(key, [])[:top_k]
        return [SearchHit(recipe_id, float(len(ids) - i)) for i, recipe_id in enumerate(ids)]


def test_recall_and_mrr_are_computed_with_ranx():
    retriever = StubRetriever({"chicken rice": ["r2", "r1"]})
    queries = [{"query_id": "q1", "ingredients": ["chicken", "rice"]}]
    qrels = {"q1": {"r1": 1}}
    results = evaluate_retriever(retriever, queries, qrels, top_k=10)
    assert set(METRICS) <= set(results)
    assert results["recall@10"] == pytest.approx(1.0)
    assert results["mrr@10"] == pytest.approx(0.5)
    assert results["n_queries"] == 1


def test_a_query_without_hits_scores_zero_and_is_not_skipped():
    """A query with no hit is an empty result (scored 0), not a skipped query."""
    retriever = StubRetriever({"chicken rice": ["r1"]})
    queries = [
        {"query_id": "q1", "ingredients": ["chicken", "rice"]},
        {"query_id": "q2", "ingredients": ["nothing"]},
    ]
    qrels = {"q1": {"r1": 1}, "q2": {"r2": 1}}
    results = evaluate_retriever(retriever, queries, qrels, top_k=10)
    assert results["n_queries"] == 2
    assert results["recall@10"] == pytest.approx(0.5)
    assert results["mrr@10"] == pytest.approx(0.5)


def test_a_run_without_any_hit_scores_zero():
    retriever = StubRetriever({})
    queries = [{"query_id": "q1", "ingredients": ["nothing"]}]
    results = evaluate_retriever(retriever, queries, {"q1": {"r1": 1}}, top_k=10)
    assert results["recall@10"] == pytest.approx(0.0)
    assert results["n_queries"] == 1


def test_load_split_reads_only_the_asked_split(tmp_path: Path):
    write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"query_id": "q1", "split": "validation", "ingredients": ["rice"]},
            {"query_id": "q2", "split": "test", "ingredients": ["salt"]},
        ],
    )
    write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {"query_id": "q1", "recipe_id": "r1", "relevance": 1},
            {"query_id": "q2", "recipe_id": "r2", "relevance": 1},
        ],
    )
    queries, qrels = load_split(tmp_path, "validation")
    assert [q["query_id"] for q in queries] == ["q1"]
    assert qrels == {"q1": {"r1": 1}}


def test_comparison_table_contains_the_measured_numbers():
    records = [
        {
            "mode": "ingredients",
            "metrics": {"recall@5": 0.5, "recall@10": 0.75, "mrr@10": 0.6, "n_queries": 10},
            "encode_seconds": 12.5,
            "cold_load_seconds": 1.2,
            "peak_rss_mb": 900.0,
        },
        {
            "mode": "title_ingredients",
            "metrics": {"recall@5": 0.6, "recall@10": 0.8, "mrr@10": 0.7, "n_queries": 10},
            "encode_seconds": 13.0,
            "cold_load_seconds": 1.3,
            "peak_rss_mb": 910.0,
        },
    ]
    table = compare(records)
    assert "| E04 |" in table and "| E05 |" in table
    assert "0.7500" in table and "0.8000" in table
    assert "12.5" in table and "13.0" in table


def test_evaluation_module_does_not_shadow_hoangs_module():
    """src.evaluate belongs to Person 1; this module must not replace it."""
    import src.evaluation as evaluation

    assert evaluation.__name__ == "src.evaluation"
    assert evaluation.METRICS == ("recall@5", "recall@10", "mrr@10")
