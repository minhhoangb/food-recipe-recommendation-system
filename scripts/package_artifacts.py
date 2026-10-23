"""Zip data/processed (pilot + main) for sharing on GitHub Release / Drive. Raw CSV is never included.

    python scripts/package_artifacts.py
Writes dist/processed-<corpus sha256 prefix>.zip and prints its sha256 (put both in
configs/default.yaml under `artifacts`, or pass them to scripts/fetch_artifacts.py).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, processed_dir  # noqa: E402
from src.utils import REPO_ROOT, read_json, sha256_file  # noqa: E402


def main() -> int:
    cfg = load_config()
    manifest = processed_dir(cfg, "main") / "dataset_manifest.json"
    if not manifest.exists():
        print("Nothing to package: run python -m src.data_collection first.")
        return 1
    prefix = read_json(manifest)["files"]["corpus.jsonl"][:10]
    archive = shutil.make_archive(
        str(REPO_ROOT / "dist" / f"processed-{prefix}"), "zip", root_dir=processed_dir(cfg, "main").parent
    )
    print(f"{archive}  ({Path(archive).stat().st_size / 1e6:.1f} MB)\nsha256: {sha256_file(archive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
