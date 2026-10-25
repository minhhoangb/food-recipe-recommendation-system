"""scikit-learn pipelines: text vectorisers for the baselines and the recipe feature table."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, OneHotEncoder, StandardScaler

from src.features import MeanIngredientRarity, identity, ingredient_count
from src.normalize import normalize_ingredients

SEED = 42


def normalize_all(recipes: list[list[str]]) -> list[list[str]]:
    """Same normalisation for every document and every query (module level so it can be pickled).

    Corpus ingredients were already normalised once by the data pipeline; a second pass changes
    ~0.6% of the distinct names (see reports/data_audit.md). That is harmless for retrieval because
    documents and queries both go through exactly this step, so identical inputs stay identical.
    """
    return [normalize_ingredients(recipe) for recipe in recipes]


def _normalizer() -> FunctionTransformer:
    return FunctionTransformer(normalize_all)


def build_text_pipeline(**tfidf_params) -> Pipeline:
    """normalise -> TF-IDF over ingredient names. tfidf_params are tuned later with GridSearchCV."""
    tfidf = TfidfVectorizer(analyzer=identity, **tfidf_params)
    return Pipeline([("normalize", _normalizer()), ("tfidf", tfidf)])


def build_lsa_pipeline(n_components: int = 256, **tfidf_params) -> Pipeline:
    """normalise -> TF-IDF -> TruncatedSVD (LSA) -> L2 normalise, so cosine = dot product."""
    return Pipeline(
        [
            ("normalize", _normalizer()),
            ("tfidf", TfidfVectorizer(analyzer=identity, **tfidf_params)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=SEED)),
            ("l2", Normalizer(copy=False)),
        ]
    )


def build_feature_pipeline() -> ColumnTransformer:
    """Recipe feature table: scaled numeric features + one-hot source (missing source -> "unknown").

    Input is the frame from `recipes_to_frame`. Missing values: the data pipeline drops recipes with
    missing text fields, so only `source` can be missing and it gets an explicit "unknown" category.
    Scaling: StandardScaler, because counts (~8) and mean IDF (~5) have different ranges and the
    features are meant for distance-based rerank/analysis.
    """
    numeric = Pipeline(
        [
            ("normalize", _normalizer()),
            (
                "features",
                FeatureUnion(
                    [("count", FunctionTransformer(ingredient_count)), ("rarity", MeanIngredientRarity())]
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    source = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("numeric", numeric, "ingredients"), ("source", source, ["source"])])


def recipes_to_frame(records: list[dict]) -> pd.DataFrame:
    """Corpus records -> the DataFrame expected by build_feature_pipeline."""
    return pd.DataFrame(
        {
            "ingredients": [r["ingredients_normalized"] for r in records],
            "source": [r["source"] or None for r in records],
        }
    )
