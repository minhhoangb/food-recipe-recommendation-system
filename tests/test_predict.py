"""Tests for `recommend()`: matched/missing ingredients and the E06 rerank."""

from __future__ import annotations

import pytest

from src.contracts import SearchHit
from src.predict import (
    MissingFirstRetriever,
    load_corpus_metadata,
    recommend,
    rerank_missing_first,
    split_ingredients,
)

METADATA = {
    "r1": {
        "title": "Chicken Rice",
        "directions": ["Cook"],
        "link": "https://example.com/r1",
        "ingredients_normalized": ["chicken", "rice", "soy sauce"],
    },
    "r2": {
        "title": "Rice Only",
        "directions": [],
        "link": "https://example.com/r2",
        "ingredients_normalized": ["rice"],
    },
    "r3": {
        "title": "Fancy",
        "directions": [],
        "link": "https://example.com/r3",
        "ingredients_normalized": ["chicken", "rice", "ginger", "star anise"],
    },
}


class StubRetriever:
    """Returns the hits in a fixed order; ignores the query."""

    name = "stub"
    version = "1"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, ingredients, top_k):
        self.calls.append((list(ingredients), top_k))
        return list(self.hits[:top_k])


def test_matched_and_missing_are_split_on_the_normalised_sets():
    matched, missing = split_ingredients(["chicken", "rice"], ["rice", "soy sauce", "chicken"])
    assert matched == ["chicken", "rice"]
    assert missing == ["soy sauce"]


def test_recommend_joins_the_metadata_and_computes_matched_and_missing():
    retriever = StubRetriever([SearchHit("r1", 0.9), SearchHit("r2", 0.5)])
    results = recommend(["chicken", "rice"], 2, retriever=retriever, metadata=METADATA)
    assert [result.recipe_id for result in results] == ["r1", "r2"]
    assert results[0].title == "Chicken Rice"
    assert results[0].matched_ingredients == ["chicken", "rice"]
    assert results[0].missing_ingredients == ["soy sauce"]
    assert results[0].directions == ["Cook"]
    assert results[1].matched_ingredients == ["rice"]
    assert results[1].missing_ingredients == []


def test_recommend_normalises_the_query_before_searching():
    retriever = StubRetriever([])
    recommend(["1 cup chicken", "RICE"], 3, retriever=retriever, metadata=METADATA)
    assert retriever.calls[0][0] == ["chicken", "rice"]


def test_recommend_respects_top_k():
    retriever = StubRetriever([SearchHit("r1", 0.9), SearchHit("r2", 0.5), SearchHit("r3", 0.1)])
    results = recommend(["rice"], 2, retriever=retriever, metadata=METADATA)
    assert len(results) == 2


def test_an_empty_query_after_normalisation_returns_nothing():
    retriever = StubRetriever([SearchHit("r1", 0.9)])
    assert recommend(["!!!", "  "], 5, retriever=retriever, metadata=METADATA) == []
    assert retriever.calls == []


def test_the_row_order_of_the_corpus_does_not_shift_the_result_rows():
    """The results must line up with the hits, whatever the metadata order is."""
    shuffled = {"r2": METADATA["r2"], "r1": METADATA["r1"]}
    retriever = StubRetriever([SearchHit("r1", 0.9), SearchHit("r2", 0.5)])
    results = recommend(["chicken", "rice"], 2, retriever=retriever, metadata=shuffled)
    assert [(r.recipe_id, r.title) for r in results] == [("r1", "Chicken Rice"), ("r2", "Rice Only")]


def test_missing_first_rerank_puts_the_recipe_with_fewest_missing_first():
    retriever = StubRetriever([SearchHit("r3", 0.9), SearchHit("r2", 0.5), SearchHit("r1", 0.1)])
    results = recommend(
        ["chicken", "rice"], 3, retriever=retriever, metadata=METADATA, rerank="missing_first"
    )
    assert [result.recipe_id for result in results] == ["r2", "r1", "r3"]
    assert results[0].missing_ingredients == []


def test_missing_first_is_off_by_default():
    retriever = StubRetriever([SearchHit("r3", 0.9), SearchHit("r2", 0.5)])
    results = recommend(["chicken", "rice"], 2, retriever=retriever, metadata=METADATA)
    assert [result.recipe_id for result in results] == ["r3", "r2"]
    assert retriever.calls[0][1] == 2, "without the rerank only top_k candidates are fetched"


def test_missing_first_asks_the_retriever_for_the_top_100_candidates():
    retriever = StubRetriever([])
    recommend(
        ["chicken", "rice"], 5, retriever=retriever, metadata=METADATA, rerank="missing_first"
    )
    assert retriever.calls[0][1] == 100


def test_the_rerank_breaks_ties_by_score_then_recipe_id():
    """Two recipes with the same number of missing ingredients are ordered by score, then id."""
    metadata = {
        "r1": {"ingredients_normalized": ["rice", "salt"]},
        "r2": {"ingredients_normalized": ["rice", "pepper"]},
    }
    hits = [SearchHit("r2", 0.5), SearchHit("r1", 0.5)]
    ordered = rerank_missing_first(hits, ["rice"], metadata)
    assert [hit.recipe_id for hit in ordered] == ["r1", "r2"]

    ordered = rerank_missing_first([SearchHit("r2", 0.9), SearchHit("r1", 0.5)], ["rice"], metadata)
    assert [hit.recipe_id for hit in ordered] == ["r2", "r1"], "the score breaks the tie"


def test_the_ablation_retriever_is_a_retriever_for_the_evaluation_driver():
    retriever = MissingFirstRetriever(StubRetriever([SearchHit("r3", 0.9), SearchHit("r2", 0.5)]), METADATA)
    hits = retriever.search(["chicken", "rice"], 2)
    assert [hit.recipe_id for hit in hits] == ["r2", "r3"]


def test_load_corpus_metadata_reads_the_fields_the_api_returns(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"recipe_id": "r1", "title": "T", "directions": ["d"], "link": "https://x", '
        '"ingredients_normalized": ["rice"]}\n',
        encoding="utf-8",
    )
    metadata = load_corpus_metadata(corpus)
    assert metadata["r1"]["title"] == "T"
    assert metadata["r1"]["ingredients_normalized"] == ["rice"]


def test_an_unknown_recipe_id_does_not_crash_the_join():
    retriever = StubRetriever([SearchHit("nope", 0.9)])
    results = recommend(["rice"], 1, retriever=retriever, metadata=METADATA)
    assert results[0].recipe_id == "nope"
    assert results[0].title == ""
    assert results[0].matched_ingredients == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_returns_nothing(top_k):
    retriever = StubRetriever([SearchHit("r1", 0.9)])
    assert recommend(["rice"], top_k, retriever=retriever, metadata=METADATA) == []
