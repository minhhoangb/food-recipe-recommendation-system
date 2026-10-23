from src.config import processed_dir
from src.reference import TfidfReference
from src.utils import read_jsonl
from tests.contract_utils import assert_retriever_contract


def _retriever_and_ids(built_fixture):
    cfg, _, _ = built_fixture
    corpus = processed_dir(cfg, "main") / "corpus.jsonl"
    return TfidfReference.from_corpus(corpus), {r["recipe_id"] for r in read_jsonl(corpus)}


def test_reference_follows_retriever_contract(built_fixture):
    retriever, ids = _retriever_and_ids(built_fixture)
    assert_retriever_contract(
        retriever, ids, [["garlic", "rice"], ["egg", "milk", "butter"], ["unobtainium"]]
    )


def test_reference_ranking_ties_and_zero_overlap():
    retriever = TfidfReference({"b": ["rice", "egg"], "a": ["rice", "egg"], "c": ["rice"], "d": ["milk"]})
    hits = retriever.search(["rice", "egg"], 10)
    assert [h.recipe_id for h in hits] == ["a", "b", "c"]  # equal scores -> recipe_id ascending
    assert hits[0].score == hits[1].score > hits[2].score > 0
    assert retriever.search(["unobtainium"], 5) == []  # zero overlap -> no arbitrary results
    assert retriever.search(["rice"], 0) == []
