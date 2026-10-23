"""Config = configs/default.yaml, optionally merged with a profile file and overrides (OmegaConf)."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils import REPO_ROOT, sha256_obj

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
RAW_ENV_VAR = "RECIPENLG_RAW"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> DictConfig:
    """Read it as cfg.data.n_main or cfg["data"]["n_main"]. `path` only needs the keys it changes."""
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if path:
        profile_file = Path(path) if Path(path).exists() else REPO_ROOT / path
        if profile_file.resolve() != DEFAULT_CONFIG.resolve():
            cfg = OmegaConf.merge(cfg, OmegaConf.load(profile_file))
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def repo_path(value: str | Path) -> Path:
    """Configured paths are relative to the repo root unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def processed_dir(cfg: DictConfig, profile: str | None = None) -> Path:
    return repo_path(cfg.paths.processed_dir) / (profile or cfg.data.profile)


def raw_csv(cfg: DictConfig, override: str | Path | None = None) -> Path:
    """CLI argument > env RECIPENLG_RAW > config paths.raw_csv."""
    return Path(override or os.environ.get(RAW_ENV_VAR) or repo_path(cfg.paths.raw_csv))


def fingerprint(cfg: DictConfig) -> str:
    return sha256_obj(OmegaConf.to_container(cfg, resolve=True))
