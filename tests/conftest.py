from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.data_collection import build

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "raw_sample.csv"

SMALL_OVERRIDES = {
    "data": {"n_main": 25, "n_pilot": 10, "chunk_rows": 7},
    "queries": {
        "min_recipe_ingredients": 3,
        "common_df_threshold": 0.3,
        "caps": {
            "train": {"main": 10, "pilot": 5},
            "validation": {"main": 4, "pilot": 2},
            "test": {"main": 6, "pilot": 3},
        },
    },
}


def small_config(processed_dir: Path, **data_overrides):
    overrides = {
        **SMALL_OVERRIDES,
        "data": {**SMALL_OVERRIDES["data"], **data_overrides},
        "paths": {"processed_dir": str(processed_dir)},
    }
    return load_config("configs/default.yaml", overrides)


@pytest.fixture(scope="session")
def fixture_csv() -> Path:
    return FIXTURE_CSV


@pytest.fixture(scope="session")
def built_fixture(tmp_path_factory, fixture_csv):
    """Run the real data pipeline once on the tiny fixture CSV."""
    out = tmp_path_factory.mktemp("processed")
    cfg = small_config(out)
    manifests = build(cfg, fixture_csv, ["pilot", "main"])
    return cfg, manifests, out
