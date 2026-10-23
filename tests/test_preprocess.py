import math

import pytest

from src.preprocess import build_recipe, normalize_link, parse_str_list


def _row(**overrides):
    row = {
        "title": "  Garlic   Rice ",
        "ingredients": '["1 cup rice", "2 cloves garlic", "1 tbsp soy sauce"]',
        "directions": '["Cook.", "Serve."]',
        "link": "www.example.com/r/1",
        "source": "Gathered",
        "NER": '["rice", "garlic", "soy sauce"]',
    }
    row.update(overrides)
    return row


def test_parse_json_and_python_literal_lists():
    assert parse_str_list('["a", " b  c "]') == ["a", "b c"]
    assert parse_str_list("['a', 'b']") == ["a", "b"]


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not a list", "[1, 2]", '{"a": 1}', "__import__('os').system('echo x')", None, math.nan],
)
def test_parse_rejects_malformed_without_executing(bad):
    assert parse_str_list(bad) is None


def test_link_gets_https_scheme_and_rejects_junk():
    assert normalize_link("www.cookbooks.com/x?id=1") == "https://www.cookbooks.com/x?id=1"
    assert normalize_link("http://a.com/x") == "http://a.com/x"
    assert normalize_link("javascript://alert(1)") == ""
    assert normalize_link("ftp://a.com") == ""
    assert normalize_link("has space.com") == ""
    assert normalize_link(math.nan) == ""


def test_build_recipe_ok():
    record, reason = build_recipe(7, _row())
    assert reason is None
    assert record["recipe_id"] == "recipenlg:7"
    assert record["title"] == "Garlic Rice"
    assert record["ingredients_normalized"] == ["rice", "garlic", "soy sauce"]
    assert record["link"] == "https://www.example.com/r/1"
    assert record["ingredients_raw"][0] == "1 cup rice"  # raw text is kept verbatim
    assert record["directions"] == ["Cook.", "Serve."]  # directions are never rewritten


def test_empty_ner_falls_back_to_raw_lines():
    record, reason = build_recipe(1, _row(NER="[]"))
    assert reason is None
    assert record["ingredients_normalized"] == ["rice", "garlic", "soy sauce"]


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"ingredients": "oops"}, "malformed_ingredients"),
        ({"directions": "oops"}, "malformed_directions"),
        ({"directions": "[]"}, "empty_directions"),
        ({"NER": "oops"}, "malformed_ner"),
        ({"title": math.nan}, "missing_field"),
        ({"title": "   "}, "empty_title"),
        ({"link": "ftp://x"}, "missing_link"),
        ({"ingredients": '["1 cup rice"]', "NER": '["rice"]'}, "ingredient_count_out_of_range"),
    ],
)
def test_drop_reasons(overrides, reason):
    record, got = build_recipe(1, _row(**overrides))
    assert record is None and got == reason
