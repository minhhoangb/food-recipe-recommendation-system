from __future__ import annotations

from pathlib import Path

import pytest

from src.build_pairs import NoDuplicateBatchSampler, build_pairs, load_pairs, save_pairs_and_manifest
from src.utils import read_json, sha256_file, write_json, write_jsonl


@pytest.fixture
def mock_recipe_dataset(tmp_path: Path) -> Path:
    """Fixture with train, validation, and test recipes across distinct groups."""
    corpus = [
        # Train recipes
        {
            "recipe_id": "recipenlg:1",
            "group_id": "grp:1",
            "title": "Tomato Basil Soup",
            "ingredients_raw": ["tomatoes", "fresh basil", "garlic", "olive oil"],
            "ingredients_normalized": ["tomato", "basil", "garlic", "olive oil"],
            "directions": ["Simmer."],
            "link": "https://example.com/1",
            "source": "Gathered",
            "ner_raw": ["tomato", "basil", "garlic", "olive oil"],
        },
        {
            "recipe_id": "recipenlg:2",
            "group_id": "grp:2",
            "title": "Garlic Bread Delight",
            "ingredients_raw": ["baguette", "garlic", "butter", "parsley"],
            "ingredients_normalized": ["baguette", "garlic", "butter", "parsley"],
            "directions": ["Toast."],
            "link": "https://example.com/2",
            "source": "Gathered",
            "ner_raw": ["baguette", "garlic", "butter", "parsley"],
        },
        {
            "recipe_id": "recipenlg:3",
            "group_id": "grp:3",
            "title": "Basil Pesto Pasta",
            "ingredients_raw": ["pasta", "basil", "pine nuts", "parmesan"],
            "ingredients_normalized": ["pasta", "basil", "pine nut", "parmesan"],
            "directions": ["Toss."],
            "link": "https://example.com/3",
            "source": "Gathered",
            "ner_raw": ["pasta", "basil", "pine nut", "parmesan"],
        },
        # Validation recipe (must NEVER be in train pairs)
        {
            "recipe_id": "recipenlg:4",
            "group_id": "grp:4",
            "title": "Validation Secret Stew",
            "ingredients_raw": ["beef", "carrots", "potatoes"],
            "ingredients_normalized": ["beef", "carrot", "potato"],
            "directions": ["Boil."],
            "link": "https://example.com/4",
            "source": "Gathered",
            "ner_raw": ["beef", "carrot", "potato"],
        },
        # Test recipe (must NEVER be in train pairs)
        {
            "recipe_id": "recipenlg:5",
            "group_id": "grp:5",
            "title": "Test Forbidden Cake",
            "ingredients_raw": ["flour", "sugar", "cocoa powder"],
            "ingredients_normalized": ["flour", "sugar", "cocoa powder"],
            "directions": ["Bake."],
            "link": "https://example.com/5",
            "source": "Gathered",
            "ner_raw": ["flour", "sugar", "cocoa powder"],
        },
    ]
    write_jsonl(tmp_path / "corpus.jsonl", corpus)

    splits = {
        "train": ["recipenlg:1", "recipenlg:2", "recipenlg:3"],
        "validation": ["recipenlg:4"],
        "test": ["recipenlg:5"],
    }
    write_json(tmp_path / "splits.json", splits)
    return tmp_path


def test_no_pair_from_validation_or_test(mock_recipe_dataset: Path):
    """DoD: không cặp nào lấy từ recipe validation hoặc test."""
    pairs, stats = build_pairs(mock_recipe_dataset, split="train", max_pairs=50, seed=42)

    splits = read_json(mock_recipe_dataset / "splits.json")
    val_set = set(splits["validation"])
    test_set = set(splits["test"])
    train_set = set(splits["train"])

    assert len(pairs) > 0
    for pair in pairs:
        rid = pair["recipe_id"]
        assert rid not in val_set, f"Leaked validation recipe {rid} into train pairs!"
        assert rid not in test_set, f"Leaked test recipe {rid} into train pairs!"
        assert rid in train_set


def test_batch_sampler_has_no_duplicate_recipe_or_group():
    """DoD: batch không có hai cặp cùng recipe hoặc cùng nhóm."""
    sample_pairs = [
        {"recipe_id": "r1", "group_id": "g1", "query": "q1", "positive": "p1"},
        {"recipe_id": "r1", "group_id": "g1", "query": "q2", "positive": "p1"},
        {"recipe_id": "r2", "group_id": "g2", "query": "q3", "positive": "p2"},
        {"recipe_id": "r3", "group_id": "g2", "query": "q4", "positive": "p3"},  # shares group g2
        {"recipe_id": "r4", "group_id": "g3", "query": "q5", "positive": "p4"},
        {"recipe_id": "r5", "group_id": "g4", "query": "q6", "positive": "p5"},
    ]

    batch_size = 3
    sampler = NoDuplicateBatchSampler(sample_pairs, batch_size=batch_size, seed=42, drop_last=False)

    batches = list(sampler)
    assert len(batches) > 0

    for batch in batches:
        recipes_in_batch = [sample_pairs[idx]["recipe_id"] for idx in batch]
        groups_in_batch = [sample_pairs[idx]["group_id"] for idx in batch]

        # Assert no two pairs share the same recipe
        assert len(recipes_in_batch) == len(set(recipes_in_batch)), f"Duplicate recipe in batch: {recipes_in_batch}"
        # Assert no two pairs share the same group
        assert len(groups_in_batch) == len(set(groups_in_batch)), f"Duplicate group in batch: {groups_in_batch}"


def test_pairs_manifest_and_hash(mock_recipe_dataset: Path, tmp_path: Path):
    """DoD: có pairs + manifest hash."""
    pairs, stats = build_pairs(mock_recipe_dataset, split="train", max_pairs=10, seed=42)
    pairs_file, manifest_file = save_pairs_and_manifest(
        pairs,
        stats,
        mock_recipe_dataset,
        out_dir=tmp_path,
        profile="pilot",
    )

    assert pairs_file.exists()
    assert manifest_file.exists()

    manifest = read_json(manifest_file)
    assert manifest["pairs_sha256"] == sha256_file(pairs_file)
    assert manifest["n_pairs"] == len(pairs)
    assert manifest["split"] == "train"


def test_positive_formatting_modes(mock_recipe_dataset: Path):
    """Verify positive formatting matches Hưng's format_document for E04 & E05."""
    pairs_e04, _ = build_pairs(mock_recipe_dataset, split="train", mode="ingredients", max_pairs=1, seed=42)
    pairs_e05, _ = build_pairs(mock_recipe_dataset, split="train", mode="title_ingredients", max_pairs=1, seed=42)

    # E04 ingredients only (no title)
    assert ". Ingredients:" not in pairs_e04[0]["positive"]
    # E05 title + ingredients
    assert ". Ingredients:" in pairs_e05[0]["positive"]
