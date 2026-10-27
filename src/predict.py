"""recommend(): normalise -> retrieve -> join metadata -> matched/missing ingredients.

    from src.predict import recommend, load_corpus_metadata
    metadata = load_corpus_metadata("data/processed/main/corpus.jsonl")
    results = recommend(["chicken", "rice"], 5, retriever=retriever, metadata=metadata)

`matched_ingredients` are the query ingredients the recipe uses, `missing_ingredients` the
recipe ingredients the user does not have. Both are computed on the *normalised* ingredient
sets, so they never depend on the order of the rows of the corpus.

E06 ablation: `MissingFirstRetriever` reranks the top 100 candidates by the number of
missing ingredients (fewest missing first). It is off by default (`rerank="none"`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from src.contracts import RecipeResult, SearchHit
from src.normalize import normalize_ingredients
from src.utils import read_jsonl

Rerank = Literal["none", "missing_first"]
DEFAULT_RERANK_CANDIDATES = 100


def load_corpus_metadata(corpus_jsonl: str | Path) -> dict[str, dict]:
    """recipe_id -> the fields the API returns (title, directions, link, ingredients)."""
    metadata: dict[str, dict] = {}
    for row in read_jsonl(corpus_jsonl):
        metadata[row["recipe_id"]] = {
            "title": row["title"],
            "directions": row["directions"],
            "link": row["link"],
            "ingredients_normalized": row["ingredients_normalized"],
        }
    return metadata


def split_ingredients(query: list[str], recipe: list[str]) -> tuple[list[str], list[str]]:
    """(matched, missing) of the query against one recipe, both normalised and deduplicated."""
    recipe_set = set(recipe)
    query_set = set(query)
    matched = [name for name in query if name in recipe_set]
    missing = [name for name in recipe if name not in query_set]
    return list(dict.fromkeys(matched)), list(dict.fromkeys(missing))


def missing_count(hit: SearchHit, query: list[str], metadata: dict[str, dict]) -> int:
    """How many of the recipe's ingredients the user does not have (unknown recipe -> 0)."""
    recipe = metadata.get(hit.recipe_id, {}).get("ingredients_normalized", [])
    query_set = set(query)
    return sum(1 for name in recipe if name not in query_set)


def rerank_missing_first(
    hits: list[SearchHit], query: list[str], metadata: dict[str, dict]
) -> list[SearchHit]:
    """Fewest missing ingredients first; the retriever score only breaks ties."""
    return sorted(
        hits,
        key=lambda hit: (missing_count(hit, query, metadata), -hit.score, hit.recipe_id),
    )


class MissingFirstRetriever:
    """E06 ablation: rerank the top candidates by the number of missing ingredients.

    It has the shape of a `Retriever` (name, version, search) so the evaluation driver can
    score it like any other model, but the order is *not* the raw score order, so it
    breaks the "score descending" clause of the contract on purpose.
    """

    name = "minilm_pretrained-missing-first"
    version = "1"

    def __init__(
        self,
        retriever: Any,
        metadata: dict[str, dict],
        candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        self._retriever = retriever
        self._metadata = metadata
        self._candidates = candidates

    def search(self, ingredients: list[str], top_k: int) -> list[SearchHit]:
        if not ingredients or top_k < 1:
            return []
        hits = self._retriever.search(ingredients, self._candidates)
        query = normalize_ingredients(ingredients)
        return rerank_missing_first(hits, query, self._metadata)[:top_k]


def recommend(
    ingredients: list[str],
    top_k: int = 5,
    *,
    retriever: Any,
    metadata: dict[str, dict],
    rerank: Rerank = "none",
    candidates: int = DEFAULT_RERANK_CANDIDATES,
) -> list[RecipeResult]:
    """Recommend `top_k` recipes for a raw ingredient list.

    The ingredients are normalised here (the API contract validates them too), the
    retriever returns the hits and every hit is joined with the corpus metadata.
    """
    query = normalize_ingredients(ingredients)
    if not query or top_k < 1:
        return []
    if rerank == "missing_first":
        retriever = MissingFirstRetriever(retriever, metadata, candidates)
    hits = retriever.search(query, top_k)

    results: list[RecipeResult] = []
    for hit in hits:
        recipe = metadata.get(hit.recipe_id, {})
        matched, missing = split_ingredients(query, recipe.get("ingredients_normalized", []))
        results.append(
            RecipeResult(
                recipe_id=hit.recipe_id,
                title=recipe.get("title", ""),
                score=hit.score,
                matched_ingredients=matched,
                missing_ingredients=missing,
                directions=recipe.get("directions", []),
                link=recipe.get("link", ""),
            )
        )
    return results


__all__ = [
    "DEFAULT_RERANK_CANDIDATES",
    "MissingFirstRetriever",
    "Rerank",
    "load_corpus_metadata",
    "recommend",
    "rerank_missing_first",
    "split_ingredients",
]
