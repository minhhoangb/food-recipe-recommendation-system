"""Tests for the benchmark driver: percentiles, bundle discovery and the report format."""

from __future__ import annotations

import json

import pytest

from src.benchmark import bundle_size_mb, find_bundles, machine_info, percentile


def test_the_percentiles_are_nearest_rank():
    values = [float(n) for n in range(1, 101)]  # 1..100
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 1.0) == 100.0
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.95) == 7.0


def test_the_percentile_does_not_depend_on_the_order():
    assert percentile([3.0, 1.0, 2.0], 0.50) == 2.0


def test_the_bundle_size_is_the_size_of_the_whole_folder(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1024 * 1024)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 1024 * 1024)
    assert bundle_size_mb(tmp_path) == pytest.approx(2.0, rel=1e-3)


def test_explicit_bundles_win_over_the_folder_content(tmp_path):
    (tmp_path / "bundle-a").mkdir()
    (tmp_path / "bundle-a" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bundle-b").mkdir()
    found = find_bundles(tmp_path, ["bundle-b"])
    assert [bundle.name for bundle in found] == ["bundle-b"]


def test_every_bundle_with_a_manifest_is_found(tmp_path):
    for name in ("bundle-a", "bundle-b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "not-a-bundle").mkdir()
    found = find_bundles(tmp_path, [])
    assert [bundle.name for bundle in found] == ["bundle-a", "bundle-b"]


def test_the_machine_info_names_the_cpu_and_the_ram():
    info = machine_info()
    assert info["platform"]
    assert info["processor"]
    assert info["python"]


def test_the_report_is_written_as_json(tmp_path):
    from src.utils import write_json

    out = tmp_path / "benchmark.json"
    write_json(out, {"machine": {"processor": "x"}, "bundles": []})
    assert json.loads(out.read_text(encoding="utf-8"))["bundles"] == []
