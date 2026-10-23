"""Ingredient normalisation: the single implementation used for recipes, queries and the API.

Quantities, units and preparation words come from `ingredient-parser-nlp`; plural endings from
`inflect`. Multi-word names are kept, so "chicken stock" != "chicken breast". The fingerprint covers
the rules below *and* the library versions, and is stored in every manifest.

First use of ingredient_parser downloads one small NLTK resource (needs internet once; in Docker do
`python -c "import ingredient_parser"` at build time).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from importlib.metadata import version

import inflect
from ingredient_parser import parse_ingredient

from src.utils import sha256_obj

NORMALIZATION_VERSION = "2"

# The parser keeps these inside the name; they do not change which ingredient it is.
DROP_WORDS = frozenset({"fresh", "freshly", "frozen", "large", "small", "medium", "optional"})
# Measurement/filler words that sometimes survive as a "name" (NER noise).
JUNK_NAMES = frozenset("tsp tsps tbsp tbs tbl pkg cup oz lb lbs one two no all its new lot alt".split())
ALIASES = {
    "scallion": "green onion",
    "spring onion": "green onion",
    "confectioners sugar": "powdered sugar",
    "icing sugar": "powdered sugar",
    "coriander leaf": "cilantro",
    "garbanzo bean": "chickpea",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "capsicum": "bell pepper",
}

NORMALIZATION_FINGERPRINT = sha256_obj(
    {
        "version": NORMALIZATION_VERSION,
        "drop_words": sorted(DROP_WORDS),
        "junk_names": sorted(JUNK_NAMES),
        "aliases": ALIASES,
        "ingredient_parser_nlp": version("ingredient-parser-nlp"),
        "inflect": version("inflect"),
    }
)

_singular = inflect.engine().singular_noun
_NON_LETTERS = re.compile(r"[^a-z\s]")


def _clean(name: str) -> str:
    words = [w for w in _NON_LETTERS.sub(" ", name.lower().replace("-", " ")).split() if w not in DROP_WORDS]
    if not words:
        return ""
    last = words[-1]
    if not last.endswith(("us", "ss", "is", "asses")):  # molasses, asparagus, hummus stay as they are
        words[-1] = _singular(last) or last
    cleaned = " ".join(words)
    return "" if len(cleaned) < 2 or cleaned in JUNK_NAMES else ALIASES.get(cleaned, cleaned)


@lru_cache(maxsize=200_000)
def _names(text: str) -> tuple[str, ...]:
    """All ingredient names in one line ("salt and pepper" -> salt, pepper), canonical form."""
    if not isinstance(text, str) or not text.strip():
        return ()
    return tuple(name for part in parse_ingredient(text).name if (name := _clean(part.text)))


def normalize_ingredient(text: str) -> str:
    """First canonical name in the line, or "" if nothing meaningful is left."""
    names = _names(text)
    return names[0] if names else ""


def normalize_ingredients(items: Iterable[str]) -> list[str]:
    """Normalise a list, drop empties and duplicates, keep first-seen order."""
    return list(dict.fromkeys(name for item in items for name in _names(item)))
