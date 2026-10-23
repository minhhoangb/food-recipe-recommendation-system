"""Safe parsing of RecipeNLG rows into schema-v1 recipe records (never uses eval)."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from typing import Any

from src.normalize import normalize_ingredients

_SPACES = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

DROP_REASONS = (
    "missing_field",
    "malformed_ingredients",
    "malformed_directions",
    "malformed_ner",
    "empty_title",
    "empty_directions",
    "missing_link",
    "ingredient_count_out_of_range",
    "exact_duplicate",
)


def parse_str_list(value: Any) -> list[str] | None:
    """Parse a stringified list of strings. Returns None if missing or malformed.

    JSON first, then ast.literal_eval (literals only, so it cannot execute code).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    parsed: Any
    try:
        parsed = json.loads(value)
    except ValueError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return [_SPACES.sub(" ", item).strip() for item in parsed if item.strip()]


def normalize_link(link: Any) -> str:
    """RecipeNLG links have no scheme (www.cookbooks.com/...). Return an http(s) URL or ""."""
    if not isinstance(link, str):
        return ""
    link = link.strip()
    if not link or " " in link:
        return ""
    if link.lower().startswith(("http://", "https://")):
        return link
    if "://" in link:
        return ""
    return "https://" + link


def title_key(title: str) -> str:
    return _NON_ALNUM.sub("", title.lower())


def build_recipe(
    raw_id: int,
    row: Mapping[str, Any],
    *,
    min_ingredients: int = 2,
    max_ingredients: int = 40,
) -> tuple[dict | None, str | None]:
    """Return (record, None) or (None, drop_reason). group_id is filled in later."""
    for column in ("title", "ingredients", "directions", "link", "source", "NER"):
        value = row.get(column)
        if value is None or (isinstance(value, float) and value != value):
            return None, "missing_field"

    title = _SPACES.sub(" ", str(row["title"])).strip()
    if not title:
        return None, "empty_title"
    ingredients_raw = parse_str_list(row["ingredients"])
    if not ingredients_raw:
        return None, "malformed_ingredients"
    directions = parse_str_list(row["directions"])
    if directions is None:
        return None, "malformed_directions"
    if not directions:
        return None, "empty_directions"
    ner_raw = parse_str_list(row["NER"])
    if ner_raw is None:
        return None, "malformed_ner"
    link = normalize_link(row["link"])
    if not link:
        return None, "missing_link"

    # NER names are cleaner than raw lines; fall back to the raw lines when NER is empty.
    normalized = normalize_ingredients(ner_raw) or normalize_ingredients(ingredients_raw)
    if not min_ingredients <= len(normalized) <= max_ingredients:
        return None, "ingredient_count_out_of_range"

    record = {
        "recipe_id": f"recipenlg:{raw_id}",
        "group_id": "",
        "title": title,
        "ingredients_raw": ingredients_raw,
        "ingredients_normalized": normalized,
        "directions": directions,
        "link": link,
        "source": str(row["source"]).strip(),
        "ner_raw": ner_raw,
    }
    return record, None


def raw_int_id(recipe_id: str) -> int:
    return int(recipe_id.rsplit(":", 1)[1])
