"""Checks on the real processed data. Skipped automatically when data/processed is absent."""

import pytest

from scripts.verify_data import verify_profile
from src.config import load_config, processed_dir

cfg = load_config()


@pytest.mark.slow
@pytest.mark.parametrize("profile", ["pilot", "main"])
def test_processed_profile_is_consistent(profile):
    directory = processed_dir(cfg, profile)
    if not (directory / "dataset_manifest.json").exists():
        pytest.skip(f"{directory} not built; run python -m src.data_collection")
    assert verify_profile(directory) == []
