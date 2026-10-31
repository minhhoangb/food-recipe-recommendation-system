"""Load a serving bundle of any kind, and describe it for /model-info.

A bundle folder is either lexical (`model.joblib` written by `src.baseline`) or dense
(`embeddings.npy` written by `src.embeddings`); the file that is present decides the loader,
so choosing a model is only a path in the config (`serving.bundle`):

    serving:
      bundle: models/bm25-main-1f1f4ce5

Both loaders already verify every hash the manifest declares, so a tampered bundle raises
before the app can serve a single request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import read_json, sha256_file


class BundleError(RuntimeError):
    """The bundle folder is neither lexical nor dense, or does not match its manifest."""


def bundle_kind(bundle: str | Path) -> str:
    """Which kind of bundle it is: "lexical" (model.joblib) or "dense" (embeddings.npy)."""
    directory = Path(bundle)
    if (directory / "model.joblib").exists():
        return "lexical"
    if (directory / "embeddings.npy").exists():
        return "dense"
    raise BundleError(f"{directory} has neither model.joblib nor embeddings.npy")


def load_retriever(bundle: str | Path) -> Any:
    """The retriever of a bundle folder, whichever kind it is."""
    directory = Path(bundle)
    if bundle_kind(directory) == "lexical":
        from src.baseline import load_bundle

        return load_bundle(directory)[0]
    from src.retrieval import DenseRetriever

    return DenseRetriever.from_bundle(directory)


def bundle_manifest(bundle: str | Path) -> dict:
    """The raw manifest.json of a bundle folder."""
    manifest_path = Path(bundle) / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"{manifest_path} is missing")
    return read_json(manifest_path)


def check_bundle_corpus(bundle: str | Path, corpus_path: str | Path) -> None:
    """Raise when the bundle was built from another corpus than the one being served.

    The result ids of a bundle are positions in its own corpus, so serving it over a
    different corpus silently returns the wrong recipes.
    """
    manifest = bundle_manifest(bundle)
    expected = sha256_file(corpus_path)
    if manifest.get("corpus_sha256") != expected:
        raise BundleError(
            f"{bundle} was built from another corpus than {corpus_path} "
            f"({manifest.get('corpus_sha256', '?')[:12]} != {expected[:12]}); "
            "point serving.bundle at a bundle of this corpus"
        )


def bundle_metrics(bundle: str | Path, tracking_uri: str = "sqlite:///mlflow.db") -> dict | None:
    """Validation metrics of the run that produced this bundle, or None if they cannot be found.

    Lexical bundles carry `metrics.json` (fit info only); dense bundles are scored in MLflow,
    so the run whose `bundle` param equals this run_id is looked up. Any failure (no MLflow,
    no mlflow.db, no matching run) is not an error: /model-info just reports null.
    """
    directory = Path(bundle)
    run_id = directory.name
    scored = directory / "metrics.json"
    if scored.exists():
        metrics = read_json(scored)
        if any("@" in key for key in metrics):
            return metrics
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        found = mlflow.search_runs(search_all_experiments=True, filter_string=f"params.bundle = '{run_id}'")
        if found.empty:
            return None
        row = found.iloc[0]
        return {
            key.replace("metrics.", "").replace("_at_", "@"): float(row[key])
            for key in found.columns
            if key.startswith("metrics.") and row[key] == row[key]
        }
    except Exception:
        return None
