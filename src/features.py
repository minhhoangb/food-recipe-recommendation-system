"""Recipe-level features (course Phase 2a). Learned parameters are fitted on train recipes only."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def identity(ingredients: list[str]) -> list[str]:
    """Analyzer for vectorizers: a recipe is already a list of ingredient 'words' (picklable)."""
    return ingredients


def ingredient_count(recipes: list[list[str]]) -> np.ndarray:
    """Number of distinct ingredients, shape (n, 1).

    Why: a 3-ingredient recipe and a 20-ingredient recipe overlap very differently with a short
    query, so length explains a lot of the ranking behaviour and is a natural rerank/analysis signal.
    """
    return np.array([[len(recipe)] for recipe in recipes], dtype=float)


class MeanIngredientRarity(BaseEstimator, TransformerMixin):
    """Mean IDF of a recipe's ingredients (higher = more unusual recipe), shape (n, 1).

    Why: separates "salt, sugar, flour" recipes from ones built on rare ingredients, which is what a
    user query with a rare ingredient pins down. IDF is learned on the fitted (train) recipes only;
    an ingredient never seen in fit gets the maximum IDF instead of leaking its held-out frequency.
    """

    def fit(self, recipes, y=None):
        self.vectorizer_ = TfidfVectorizer(analyzer=identity, norm=None).fit(recipes)
        self.max_idf_ = float(self.vectorizer_.idf_.max())
        return self

    def transform(self, recipes):
        sizes = np.array([len(recipe) for recipe in recipes], dtype=float)
        weights = self.vectorizer_.transform(recipes)  # each ingredient has tf=1, so value = its IDF
        known = np.asarray((weights > 0).sum(axis=1)).ravel()
        total = np.asarray(weights.sum(axis=1)).ravel() + (sizes - known) * self.max_idf_
        return (total / np.maximum(sizes, 1)).reshape(-1, 1)
