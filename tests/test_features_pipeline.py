import joblib
import numpy as np
import pytest

from src.config import processed_dir
from src.features import MeanIngredientRarity, ingredient_count
from src.pipeline import build_feature_pipeline, build_text_pipeline, recipes_to_frame
from src.utils import read_json, read_jsonl


def test_ingredient_count():
    assert ingredient_count([["a", "b"], ["c"], []]).tolist() == [[2.0], [1.0], [0.0]]


def test_mean_rarity_matches_hand_computed_idf():
    train = [["salt", "egg"], ["salt", "rice"], ["salt", "fish"]]
    # sklearn smooth idf = ln((1 + n) / (1 + df)) + 1 : salt (df=3) -> 1.0, others (df=1) -> ln(2) + 1
    rare = np.log(2) + 1
    result = MeanIngredientRarity().fit(train).transform([["salt", "egg"], ["egg"], []])
    assert result.ravel() == pytest.approx([(1 + rare) / 2, rare, 0.0])


def test_rarity_is_fit_on_train_only_and_unseen_gets_max_idf():
    train = [["salt", "egg"], ["salt", "rice"], ["salt", "fish"]]
    heldout = [["salt", "truffle"], ["salt", "truffle"], ["truffle"]]
    model = MeanIngredientRarity().fit(train)
    assert "truffle" not in model.vectorizer_.vocabulary_  # held-out vocabulary never enters fit
    rare = np.log(2) + 1
    assert model.transform(heldout[:1]).item() == pytest.approx((1 + rare) / 2)  # unseen -> max idf
    refit_on_everything = MeanIngredientRarity().fit(train + heldout)
    assert refit_on_everything.transform(train[:1]).item() != pytest.approx(
        model.transform(train[:1]).item()
    )  # proves that fitting on held-out data would have changed the feature


def _records_and_train_mask(built_fixture):
    cfg, _, _ = built_fixture
    directory = processed_dir(cfg, "main")
    records = list(read_jsonl(directory / "corpus.jsonl"))
    train_ids = set(read_json(directory / "splits.json")["train"])
    return records, np.array([r["recipe_id"] in train_ids for r in records])


def test_feature_pipeline_fit_transform_and_pickle(built_fixture, tmp_path):
    records, is_train = _records_and_train_mask(built_fixture)
    frame = recipes_to_frame(records)
    pipeline = build_feature_pipeline().fit(frame[is_train])
    features = pipeline.transform(frame)
    assert features.shape[0] == len(records) and features.shape[1] >= 3  # count, rarity, source columns
    assert np.isfinite(features.toarray() if hasattr(features, "toarray") else features).all()

    joblib.dump(pipeline, tmp_path / "features.joblib")
    again = joblib.load(tmp_path / "features.joblib").transform(frame)
    assert np.allclose(features.toarray() if hasattr(features, "toarray") else features,
                       again.toarray() if hasattr(again, "toarray") else again)


def test_feature_pipeline_scaling_uses_train_statistics(built_fixture):
    records, is_train = _records_and_train_mask(built_fixture)
    frame = recipes_to_frame(records)
    pipeline = build_feature_pipeline().fit(frame[is_train])
    numeric = pipeline.named_transformers_["numeric"].transform(frame[is_train]["ingredients"])
    assert numeric.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)  # standardised on train


def test_missing_source_becomes_unknown_category(built_fixture):
    records, _ = _records_and_train_mask(built_fixture)
    frame = recipes_to_frame(records)
    frame.loc[0, "source"] = None
    out = build_feature_pipeline().fit(frame).transform(frame)
    assert out.shape[0] == len(records)


def test_text_pipeline_roundtrips_through_joblib(tmp_path):
    docs = [["chicken breast", "rice"], ["egg", "milk"], ["rice", "egg"]]
    pipeline = build_text_pipeline().fit(docs)
    joblib.dump(pipeline, tmp_path / "text.joblib")
    loaded = joblib.load(tmp_path / "text.joblib")
    assert (pipeline.transform(docs) != loaded.transform(docs)).nnz == 0
