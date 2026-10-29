import numpy as np
import pytest

from src.baseline import MODELS, BundleError, fit_retriever, load_bundle, main, save_bundle
from src.config import load_config, processed_dir
from src.utils import read_json, read_jsonl, write_json
from tests.contract_utils import assert_retriever_contract

QUERIES = [["garlic", "rice"], ["egg", "milk", "butter"], ["unobtainium"]]


@pytest.fixture(scope="module")
def corpus_and_train(built_fixture):
    cfg, _, _ = built_fixture
    directory = processed_dir(cfg, "main")
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    return cfg, corpus, set(read_json(directory / "splits.json")["train"])


@pytest.fixture(scope="module")
def fitted(corpus_and_train):
    _, corpus, train_ids = corpus_and_train
    return {model: fit_retriever(model, corpus, train_ids) for model in MODELS}


@pytest.mark.parametrize("model", MODELS)
def test_retriever_follows_contract(model, corpus_and_train, fitted):
    _, corpus, _ = corpus_and_train
    retriever, info = fitted[model]
    assert_retriever_contract(retriever, {r["recipe_id"] for r in corpus}, QUERIES)
    assert info["n_docs"] == len(corpus) and info["fit_seconds"] >= 0


@pytest.mark.parametrize("model", MODELS)
def test_zero_overlap_query_returns_nothing(model, fitted):
    assert fitted[model][0].search(["unobtainium"], 5) == []


@pytest.mark.parametrize("model", MODELS)
def test_a_train_recipe_finds_itself(model, corpus_and_train, fitted):
    _, corpus, train_ids = corpus_and_train
    retriever, _ = fitted[model]
    for record in corpus:
        if record["recipe_id"] in train_ids:
            hits = retriever.search(record["ingredients_normalized"], 10)
            assert record["recipe_id"] in [h.recipe_id for h in hits], record["title"]


def test_tfidf_and_lsa_fit_on_train_only(corpus_and_train, fitted):
    _, corpus, train_ids = corpus_and_train
    train_vocab = {i for r in corpus if r["recipe_id"] in train_ids for i in r["ingredients_normalized"]}
    all_vocab = {i for r in corpus for i in r["ingredients_normalized"]}
    assert all_vocab - train_vocab, "fixture needs held-out-only ingredients for this test"
    for model in ("tfidf", "lsa"):
        vocabulary = set(fitted[model][0].pipeline["tfidf"].vocabulary_)
        assert vocabulary == train_vocab  # nothing from validation/test recipes leaked into fit


@pytest.mark.parametrize("model", MODELS)
def test_bundle_roundtrip_keeps_results(model, corpus_and_train, fitted, tmp_path):
    cfg, _, _ = corpus_and_train
    retriever, info = fitted[model]
    bundle = save_bundle(retriever, info, cfg, tmp_path)
    loaded, manifest = load_bundle(bundle)
    assert manifest.model_name == model and manifest.run_id == bundle.name
    for query in QUERIES:
        before, after = retriever.search(query, 10), loaded.search(query, 10)
        assert [h.recipe_id for h in before] == [h.recipe_id for h in after]
        assert np.allclose([h.score for h in before], [h.score for h in after])


def test_tampered_bundle_is_rejected(corpus_and_train, fitted, tmp_path):
    cfg, _, _ = corpus_and_train
    retriever, info = fitted["tfidf"]
    bundle = save_bundle(retriever, info, cfg, tmp_path)
    with open(bundle / "model.joblib", "ab") as handle:
        handle.write(b"x")
    with pytest.raises(BundleError, match="sha256"):
        load_bundle(bundle)


def test_bundle_with_other_normalisation_is_rejected(corpus_and_train, fitted, tmp_path):
    cfg, _, _ = corpus_and_train
    retriever, info = fitted["bm25"]
    bundle = save_bundle(retriever, info, cfg, tmp_path)
    manifest = read_json(bundle / "manifest.json")
    write_json(bundle / "manifest.json", {**manifest, "normalization_fingerprint": "0" * 64})
    with pytest.raises(BundleError, match="normalisation"):
        load_bundle(bundle)


@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_real_pilot_bundle_passes_contract(model, tmp_path):
    cfg = load_config("configs/pilot.yaml")
    directory = processed_dir(cfg, "pilot")
    if not (directory / "corpus.jsonl").exists():
        pytest.skip("pilot data not built")
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    retriever, info = fit_retriever(model, corpus, set(read_json(directory / "splits.json")["train"]))
    loaded, _ = load_bundle(save_bundle(retriever, info, cfg, tmp_path))
    assert_retriever_contract(loaded, {r["recipe_id"] for r in corpus}, QUERIES)


@pytest.fixture
def fixture_config(built_fixture, tmp_path):
    _, _, out = built_fixture
    config = tmp_path / "fixture.yaml"
    config.write_text(f"paths:\n  processed_dir: {out.as_posix()}\ndata:\n  profile: main\n", encoding="utf-8")
    return config


def test_cli_tuned_tfidf_uses_the_config_params(fixture_config, tmp_path):
    assert main(["--model", "tfidf", "--tuned", "--config", str(fixture_config), "--out", str(tmp_path)]) == 0
    (bundle,) = tmp_path.glob("tfidf_tuned-main-*")
    metrics = read_json(bundle / "metrics.json")
    assert set(metrics["tfidf_params"]) == {"tokens", "min_df", "sublinear_tf", "norm"}
    assert load_bundle(bundle)[1].model_name == "tfidf"


def test_cli_tuned_only_for_tfidf(fixture_config):
    with pytest.raises(SystemExit):
        main(["--model", "bm25", "--tuned", "--config", str(fixture_config)])


def test_word_tokens_change_the_vocabulary(corpus_and_train):
    _, corpus, train_ids = corpus_and_train
    words, _ = fit_retriever("tfidf", corpus, train_ids, {"tokens": "ingredient_and_words"})
    names, _ = fit_retriever("tfidf", corpus, train_ids)
    assert set(names.pipeline["tfidf"].vocabulary_) < set(words.pipeline["tfidf"].vocabulary_)
