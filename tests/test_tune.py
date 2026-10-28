import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

import src.tune as tune
from src.baseline import rank_top_k
from src.config import processed_dir
from src.features import identity, ingredient_and_words
from src.pipeline import build_text_pipeline

SMALL_GRID = {"analyzer": [identity], "min_df": [1, 2], "norm": ["l2", None]}


@pytest.fixture(scope="module")
def cv_data(built_fixture):
    cfg, _, _ = built_fixture
    return tune.load_cv_data(processed_dir(cfg, "main"))


def tokens(docs, ids, rows):
    return {token for doc, recipe_id in zip(docs, ids, strict=True) if recipe_id in rows for token in doc}


def test_fold_vocabulary_never_contains_held_out_recipes(cv_data, monkeypatch):
    ids, groups, docs, queries, qrels = cv_data
    calls = []

    class RecordingScorer(tune.RetrievalScorer):
        def __call__(self, vectorizer, X, y):
            calls.append((set(vectorizer.vocabulary_), set(y)))
            return super().__call__(vectorizer, X, y)

    monkeypatch.setattr(tune, "RetrievalScorer", RecordingScorer)
    tune.run_cv(ids, groups, docs, queries, qrels, param_grid=SMALL_GRID)

    held_out_only_seen = False
    for train_index, test_index in GroupKFold(n_splits=tune.N_FOLDS).split(docs, ids, groups):
        fit_rows, held_out = {ids[i] for i in train_index}, {ids[i] for i in test_index}
        held_out_only = tokens(docs, ids, held_out) - tokens(docs, ids, fit_rows)
        held_out_only_seen |= bool(held_out_only)
        fold_calls = [vocab for vocab, rows in calls if rows == held_out]
        assert len(fold_calls) == 4  # one held-out score per grid point
        for vocab in fold_calls:
            assert vocab <= tokens(docs, ids, fit_rows)  # fitted on the other folds only
            assert vocab.isdisjoint(held_out_only)
    assert held_out_only_seen, "fixture needs held-out-only ingredients for this test"


def test_held_out_recipe_is_still_a_candidate(cv_data):
    ids, _, docs, _, _ = cv_data
    target = 0
    vectorizer = TfidfVectorizer(analyzer=identity).fit(docs[1:])  # target recipe not in the fit
    scorer = tune.RetrievalScorer(ids, docs, [], {})
    scores = (vectorizer.transform([docs[target]]) @ vectorizer.transform(docs).T).toarray()[0]
    assert ids[target] in scorer.top_hits(scores)


def test_top_hits_matches_full_sort_with_ties():
    rng = np.random.default_rng(42)
    ids = np.array([f"r{i:03d}" for i in range(200)])
    scores = rng.choice([0.0, 0.2, 0.5, 0.9], size=200)  # many ties around the k-th score
    scorer = tune.RetrievalScorer(ids, [], [], {})
    expected = rank_top_k(ids, scores, 10)
    assert list(scorer.top_hits(scores)) == [hit.recipe_id for hit in expected]


def test_cv_results_and_e01_row(cv_data):
    ids, groups, docs, queries, qrels = cv_data
    search = tune.run_cv(ids, groups, docs, queries, qrels, param_grid=SMALL_GRID)
    frame = tune.results_frame(search)
    assert len(frame) == 4 and frame["tokens"].eq("ingredient").all()
    row = tune.summary_row(search, frame, {"analyzer": identity, "min_df": 1, "norm": None})
    assert row["norm"] is None and 0 <= row["mean_cv_mrr@10"] <= 1
    assert set(tune.config_params({**row, "sublinear_tf": False})) == {"tokens", "min_df", "sublinear_tf", "norm"}


def test_ingredient_and_words_splits_inside_names_only():
    assert ingredient_and_words(["chicken breast", "rice"]) == ["chicken breast", "rice", "chicken", "breast"]
    pipeline = build_text_pipeline(tokens="ingredient_and_words").fit([["chicken breast", "rice"]])
    assert "chicken" in pipeline["tfidf"].vocabulary_ and "breast rice" not in pipeline["tfidf"].vocabulary_
