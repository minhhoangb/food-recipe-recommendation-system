"""Locks the metric conventions we rely on. Metrics themselves come from `ranx` (not re-implemented).

Usage in the project:
    qrels = Qrels({query_id: {recipe_id: relevance}})
    run = Run({query_id: {hit.recipe_id: hit.score for hit in retriever.search(...)}})
    evaluate(qrels, run, ["recall@5", "recall@10", "mrr@10", "ndcg@10"], make_comparable=True)
`make_comparable=True` scores a query missing from the run as an empty result (default: error).
ranx's nDCG uses linear gain (use "ndcg_burges" for 2^rel - 1); state which one in the report.
Ties are ordered by ranx, not by recipe_id. Values below are computed by hand.
"""

import math

import pytest
from ranx import Qrels, Run, evaluate


def score(qrels, run, metrics):
    return evaluate(Qrels(qrels), Run(run), metrics)


def test_recall_uses_full_denominator_for_multiple_positives():
    # relevant {b, z}; only b is in the top 3 -> 1/2
    result = score({"q": {"b": 1, "z": 1}}, {"q": {"a": 3.0, "b": 2.0, "c": 1.0}}, ["recall@3"])
    assert result == pytest.approx(0.5)


def test_mrr_is_reciprocal_rank_within_cutoff():
    qrels = {"q": {"a": 1}}
    assert score(qrels, {"q": {"x": 3.0, "y": 2.0, "a": 1.0}}, ["mrr@10"]) == pytest.approx(1 / 3)
    assert score(qrels, {"q": {"x": 3.0, "y": 2.0, "a": 1.0}}, ["mrr@2"]) == 0.0


def test_ndcg_uses_linear_gain_by_default():
    # relevance a=2, b=1; ranking [b, a]. DCG = 1/log2(2) + 2/log2(3); ideal = 2/log2(2) + 1/log2(3)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    result = score({"q": {"a": 2, "b": 1}}, {"q": {"b": 2.0, "a": 1.0}}, ["ndcg@10"])
    assert result == pytest.approx(expected)


def test_unjudged_documents_count_as_not_relevant():
    result = score({"q": {"a": 1}}, {"q": {"u1": 3.0, "u2": 2.0, "a": 1.0}}, ["ndcg@10"])
    assert result == pytest.approx(1 / math.log2(4))


def test_mean_over_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 2.0, "x": 1.0}, "q2": {"x": 2.0, "b": 1.0}}
    result = score(qrels, run, ["recall@1", "mrr@10"])
    assert result["recall@1"] == pytest.approx(0.5)
    assert result["mrr@10"] == pytest.approx((1 + 0.5) / 2)
