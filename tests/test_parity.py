"""Tests for the parity check between the native run and the Docker container."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_parity import compare


def _answer(ids: list[str], scores: list[float]) -> dict:
    return {
        "results": [
            {"recipe_id": recipe_id, "score": score, "title": "t", "matched_ingredients": []}
            for recipe_id, score in zip(ids, scores, strict=True)
        ]
    }


def test_two_identical_answers_match():
    native = _answer(["r1", "r2"], [1.0, 0.5])
    docker = _answer(["r1", "r2"], [1.0, 0.5])
    result = compare(native, docker, 1e-6)
    assert result["ok"]
    assert result["same_ids"]
    assert result["max_score_diff"] == 0.0


def test_a_different_order_is_a_difference():
    native = _answer(["r1", "r2"], [1.0, 0.5])
    docker = _answer(["r2", "r1"], [1.0, 0.5])
    result = compare(native, docker, 1e-6)
    assert not result["ok"]
    assert not result["same_ids"]


def test_a_score_outside_the_tolerance_is_a_difference():
    native = _answer(["r1"], [1.0])
    docker = _answer(["r1"], [1.001])
    result = compare(native, docker, 1e-6)
    assert not result["ok"]
    assert not result["within_tolerance"]
    assert result["max_score_diff"] == pytest.approx(0.001, abs=1e-9)


def test_a_score_inside_the_tolerance_still_matches():
    native = _answer(["r1"], [1.0])
    docker = _answer(["r1"], [1.0000001])
    assert compare(native, docker, 1e-6)["ok"]


def test_two_empty_answers_match():
    assert compare({"results": []}, {"results": []}, 1e-6)["ok"]


def test_a_missing_result_is_a_difference():
    native = _answer(["r1", "r2"], [1.0, 0.5])
    docker = _answer(["r1"], [1.0])
    assert not compare(native, docker, 1e-6)["ok"]
