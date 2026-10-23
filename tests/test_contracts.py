import pytest
from pydantic import ValidationError

from src.contracts import BatchPredictRequest, PredictRequest, RecipeRecord, SearchHit


def test_valid_request_defaults():
    request = PredictRequest(ingredients=["chicken breast", "rice"])
    assert request.top_k == 5


@pytest.mark.parametrize(
    "payload",
    [
        {"ingredients": []},
        {"ingredients": ["  "]},
        {"ingredients": ["x" * 101]},
        {"ingredients": ["rice"] * 51},
        {"ingredients": ["rice"], "top_k": 0},
        {"ingredients": ["rice"], "top_k": 21},
        {"ingredients": ["rice"], "top_k": True},  # bool is not an int here
        {"ingredients": ["rice"], "top_k": "5"},
        {"ingredients": ["rice"], "top_k": 5.0},
        {"ingredients": ["2 cups"], "top_k": 5, "extra": 1},
        {"ingredients": ["(optional)"]},  # nothing left after normalisation
        {"ingredients": "rice"},
    ],
)
def test_invalid_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


def test_batch_limits():
    one = {"ingredients": ["rice"]}
    assert len(BatchPredictRequest(instances=[one] * 20).instances) == 20
    with pytest.raises(ValidationError):
        BatchPredictRequest(instances=[one] * 21)
    with pytest.raises(ValidationError):
        BatchPredictRequest(instances=[])


def test_search_hit_is_immutable():
    hit = SearchHit("recipenlg:1", 0.5)
    with pytest.raises(AttributeError):
        hit.score = 1.0  # type: ignore[misc]


def test_record_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RecipeRecord(
            recipe_id="r", group_id="g", title="t", ingredients_raw=[], ingredients_normalized=[],
            directions=[], link="https://x", source="s", ner_raw=[], surprise=1,
        )
