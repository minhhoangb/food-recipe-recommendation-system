"""Download + verify the processed dataset zip into data/processed (teammates and Colab).

    python scripts/fetch_artifacts.py            # artifacts.url / artifacts.sha256 from the config
    python scripts/fetch_artifacts.py --url <zip> --sha256 <hash>
pooch refuses to unpack anything whose sha256 does not match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pooch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from src.config import load_config, repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--sha256")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    url = args.url or OmegaConf.select(cfg, "artifacts.url")
    sha256 = args.sha256 or OmegaConf.select(cfg, "artifacts.sha256")
    if not url or not sha256:
        print(
            "No artifact URL/sha256 configured. Ask the data owner, or build locally:\n"
            "  python -m src.data_collection --raw-path <RecipeNLG_dataset.csv>"
        )
        return 1

    target = repo_path(cfg.paths.processed_dir)
    try:
        pooch.retrieve(
            url,
            known_hash=f"sha256:{sha256.lower()}",
            path=pooch.os_cache("recipe-artifacts"),
            processor=pooch.Unzip(extract_dir=str(target)),
        )
    except (OSError, ValueError) as error:  # download failure or hash mismatch
        print(f"failed, nothing extracted: {error}")
        return 1
    print(f"extracted to {target}. Now run: python scripts/verify_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
