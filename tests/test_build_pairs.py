from __future__ import annotations

from pathlib import Path

import pytest

from src.build_pairs import build_pairs, load_pairs, save_pairs
from src.utils import write_json, write_jsonl


@pytest.fixture
def mock_processed_dir(tmp_path: Path) -> Path:
    corpus = [
        {
            "recipe_id": "r1",
            "group_id": "g1",
            "title": "Tomato Soup",
            "ingredients_raw": ["1 can tomatoes", "salt"],
            "ingredients_normalized": ["tomato", "salt"],
            "directions": ["Boil."],
            "link": "https://example.com/1",
            "source": "Gathered",
            "ner_raw": ["tomato", "salt"],
        },
        {
            "recipe_id": "r2",
            "group_id": "g2",
            "title": "Garlic Bread",
            "ingredients_raw": ["bread", "garlic"],
            "ingredients_normalized": ["bread", "garlic"],
            "directions": ["Bake."],
            "link": "https://example.com/2",
            "source": "Gathered",
            "ner_raw": ["bread", "garlic"],
        },
        {
            "recipe_id": "r3",
            "group_id": "g3",
            "title": "Tomato Bread",
            "ingredients_raw": ["bread", "tomato"],
            "ingredients_normalized": ["bread", "tomato"],
            "directions": ["Toast."],
            "link": "https://example.com/3",
            "source": "Gathered",
            "ner_raw": ["bread", "tomato"],
        },
    ]
    write_jsonl(tmp_path / "corpus.jsonl", corpus)

    queries = [
        {
            "query_id": "q1",
            "split": "train",
            "ingredients": ["tomato", "salt"],
            "origin_recipe_ids": ["r1"],
            "origin_group_ids": ["g1"],
            "label_source": "synthetic",
        },
        {
            "query_id": "q2",
            "split": "validation",
            "ingredients": ["garlic"],
            "origin_recipe_ids": ["r2"],
            "origin_group_ids": ["g2"],
            "label_source": "synthetic",
        },
    ]
    write_jsonl(tmp_path / "queries.jsonl", queries)

    qrels = [
        {"query_id": "q1", "recipe_id": "r1", "relevance": 1, "label_source": "synthetic_origin"},
        {"query_id": "q2", "recipe_id": "r2", "relevance": 1, "label_source": "synthetic_origin"},
    ]
    write_jsonl(tmp_path / "qrels.jsonl", qrels)

    write_json(tmp_path / "splits.json", {"train": ["r1", "r3"], "validation": ["r2"], "test": []})
    return tmp_path


def test_build_pairs_reads_only_requested_split(mock_processed_dir: Path):
    pairs = build_pairs(mock_processed_dir, split="train", mode="ingredients")
    assert len(pairs) == 1
    assert pairs[0]["query_id"] == "q1"
    assert pairs[0]["recipe_id"] == "r1"
    assert pairs[0]["query"] == "tomato, salt"
    assert pairs[0]["positive"] == "tomato, salt"


def test_build_pairs_title_ingredients_mode(mock_processed_dir: Path):
    pairs = build_pairs(mock_processed_dir, split="train", mode="title_ingredients")
    assert len(pairs) == 1
    assert pairs[0]["positive"] == "Tomato Soup. Ingredients: tomato, salt"


def test_build_pairs_with_negatives_never_collides_with_origin(mock_processed_dir: Path):
    pairs = build_pairs(
        mock_processed_dir,
        split="train",
        mode="ingredients",
        include_negatives=True,
        hard_negatives=True,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert "negative" in pair
    assert pair["negative_recipe_id"] != "r1"
    # r3 has tomato, so lexical hard negative chooses r3
    assert pair["negative_recipe_id"] == "r3"


def test_save_and_load_pairs_roundtrip(tmp_path: Path):
    sample = [{"query": "garlic", "positive": "garlic bread", "query_id": "q1", "recipe_id": "r1"}]
    path = tmp_path / "pairs.jsonl"
    save_pairs(sample, path)
    loaded = load_pairs(path)
    assert loaded == sample
