import pytest

from src.normalize import NORMALIZATION_FINGERPRINT, normalize_ingredient, normalize_ingredients


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1 (10 1/2 oz.) cans chicken gravy", "chicken gravy"),
        ("2 Tbsp. butter or margarine", "butter"),
        ("1 1/2 c. firmly packed brown sugar", "brown sugar"),
        ("2 cloves garlic", "garlic"),
        ("1 tsp ground cloves", "ground clove"),
        ("Salt, to taste", "salt"),
        ("1 teaspoon salt to taste", "salt"),
        ("½ cup chopped fresh parsley", "parsley"),
        ("Tomatoes", "tomato"),
        ("2 chicken breasts, cut into cubes", "chicken breast"),
        ("baby spinach leaves", "baby spinach leaf"),
        ("chocolate chip cookies", "chocolate chip cookie"),
        ("scallions", "green onion"),
        ("molasses", "molasses"),
        ("1-cup sugar", "sugar"),
        ("", ""),
        ("   ", ""),
        ("o", ""),
        ("1 i", ""),
        ("2 cups", ""),
        ("tsp", ""),
    ],
)
def test_normalize_ingredient(raw, expected):
    assert normalize_ingredient(raw) == expected


def test_non_string_is_empty():
    assert normalize_ingredient(None) == ""  # type: ignore[arg-type]
    assert normalize_ingredient(5) == ""  # type: ignore[arg-type]


def test_multi_word_names_are_kept_distinct():
    names = ["chicken stock", "chicken breast", "baking powder", "baking soda", "ground beef", "beef"]
    assert len(set(normalize_ingredients(names))) == len(names)


def test_idempotent():
    for raw in ["2 cups chopped Tomatoes", "1 lb. ground beef", "cream of mushroom soup", "eggs"]:
        once = normalize_ingredient(raw)
        assert normalize_ingredient(once) == once


def test_list_dedups_and_keeps_order():
    assert normalize_ingredients(["2 Eggs", "egg", "Milk", "", "1 cup milk"]) == ["egg", "milk"]


def test_fingerprint_is_stable_hex():
    assert len(NORMALIZATION_FINGERPRINT) == 64
