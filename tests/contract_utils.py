"""Executable form of the Retriever contract. Every model must pass `assert_retriever_contract`.

    from tests.contract_utils import assert_retriever_contract
    def test_my_retriever_follows_contract(my_retriever, corpus_ids):
        assert_retriever_contract(my_retriever, corpus_ids, [["garlic", "rice"], ["no such thing"]])
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from src.contracts import Retriever, SearchHit


def assert_retriever_contract(
    retriever: Retriever,
    corpus_ids: Collection[str],
    example_queries: Sequence[list[str]],
    top_ks: Sequence[int] = (1, 5, 20),
) -> None:
    assert isinstance(retriever, Retriever), "needs name, version and search()"
    assert retriever.name and retriever.version

    for query in example_queries:
        previous: list[SearchHit] | None = None
        for top_k in top_ks:
            hits = retriever.search(list(query), top_k)
            assert isinstance(hits, list) and len(hits) <= top_k
            assert all(isinstance(h, SearchHit) for h in hits)
            ids = [h.recipe_id for h in hits]
            assert len(set(ids)) == len(ids), "duplicate recipe_id in result"
            assert set(ids) <= set(corpus_ids), "result id is not in the corpus"
            keys = [(-h.score, h.recipe_id) for h in hits]
            assert keys == sorted(keys), "must sort by score desc, ties by recipe_id asc"
            if previous is not None:  # top-k lists must be prefixes of each other
                assert ids[: len(previous)] == [h.recipe_id for h in previous][: len(ids)]
            previous = hits
        assert retriever.search(list(query), 5) == retriever.search(list(query), 5), "deterministic"

    assert retriever.search([], 5) == [], "empty query returns no hits"
