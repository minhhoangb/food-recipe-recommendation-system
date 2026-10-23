"""Reference retriever: TF-IDF cosine over normalised ingredients (scikit-learn).

Not one of the compared models. It exists so the API, UI, evaluator and monitor can be built and
tested against REAL data from day one, and as the executable example of the Retriever contract.
IDF is fit on the whole candidate corpus (transductive), which is fine for a reference; the
tuned, train-only baseline is Person 1's `src/baseline.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.contracts import SearchHit
from src.utils import read_jsonl


def _tokens(ingredients: list[str]) -> list[str]:
    return ingredients  # a recipe/query is already a list of normalised ingredient "words"


class TfidfReference:
    name = "tfidf-reference"
    version = "1"

    def __init__(self, recipes: dict[str, list[str]]):
        """recipes: recipe_id -> normalised ingredient list."""
        self._ids = np.array(sorted(recipes))  # sorted ids make the tie-break deterministic
        self._vectorizer = TfidfVectorizer(analyzer=_tokens)
        self._matrix = self._vectorizer.fit_transform([recipes[i] for i in self._ids])

    @classmethod
    def from_corpus(cls, corpus_jsonl: str | Path) -> TfidfReference:
        return cls({r["recipe_id"]: r["ingredients_normalized"] for r in read_jsonl(corpus_jsonl)})

    def search(self, ingredients: list[str], top_k: int) -> list[SearchHit]:
        if not ingredients or top_k < 1:
            return []
        scores = linear_kernel(self._vectorizer.transform([ingredients]), self._matrix).ravel()
        order = np.lexsort((self._ids, -scores))[:top_k]  # score desc, then recipe_id asc
        return [SearchHit(str(self._ids[i]), float(scores[i])) for i in order if scores[i] > 0]
