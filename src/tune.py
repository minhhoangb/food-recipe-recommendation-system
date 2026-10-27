"""E02: tune the TF-IDF baseline with GridSearchCV + GroupKFold(5) on the outer train split.

    python -m src.tune --config configs/default.yaml

Protocol (team contract, section 6):
- Rows are the outer-train recipes and groups their group_id, so a group never sits on both sides.
- GridSearchCV refits the vectorizer (vocabulary + IDF) on the four training folds only.
- Every outer-train recipe stays in the candidate corpus (transformed by the fold's vectorizer), so
  the held-out origin recipes can still be retrieved.
- Queries are the synthetic `train` queries; a query belongs to the fold holding its origin recipe.
  The score is MRR@10 (ranx); the same scorer on the training folds gives the train proxy.

Normalisation is stateless, so documents and queries are normalised once up front instead of in
every fit; the chosen values are then used as build_text_pipeline(**params) (configs: baselines).
GroupKFold without shuffle is deterministic, so no random_state is involved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from src.baseline import rank_top_k
from src.config import fingerprint, load_config, processed_dir
from src.evaluate import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, load_split, score_run
from src.features import identity
from src.normalize import normalize_ingredients
from src.pipeline import ANALYZERS, build_lsa_pipeline, normalize_all
from src.utils import REPO_ROOT, read_json, read_jsonl, sha256_file, write_json

N_FOLDS = 5
TOKENS = {analyzer: name for name, analyzer in ANALYZERS.items()}
PARAM_GRID = {
    # whole ingredient names vs. also the words inside them (the ngram_range question, see features)
    "analyzer": list(ANALYZERS.values()),
    # ignore tokens seen in fewer than min_df recipes: typos and one-off names (reports/data_audit.md)
    "min_df": [1, 2, 5],
    # log-scaled tf only matters when a token repeats inside one recipe (words mode)
    "sublinear_tf": [False, True],
    # l2 = cosine; None lets long recipes collect more score, i.e. tests length normalisation
    "norm": ["l2", None],
}
E01_PARAMS = {"analyzer": identity, "min_df": 1, "sublinear_tf": False, "norm": "l2"}  # sklearn defaults
LSA_COMPONENTS = 256  # what src.baseline uses whenever the vocabulary is larger (always on main)


class RetrievalScorer:
    """GridSearchCV scorer: MRR@10 of the queries whose origin recipe is among the scored rows."""

    def __init__(self, ids, docs, queries, qrels, top_k: int = 10, chunk: int = 512):
        self.ids, self.docs, self.queries, self.qrels = np.asarray(ids), docs, queries, qrels
        self.top_k, self.chunk = top_k, chunk

    def __call__(self, vectorizer: TfidfVectorizer, X, y) -> float:
        rows = set(y)
        # a query merged from recipes of two folds counts in both (rare: merged_same_split in manifest)
        queries = [q for q in self.queries if rows.intersection(q["origin_recipe_ids"])]
        documents = vectorizer.transform(self.docs)
        query_vectors = vectorizer.transform([q["ingredients"] for q in queries])
        run = {}
        for start in range(0, len(queries), self.chunk):
            scores = linear_kernel(query_vectors[start : start + self.chunk], documents)
            for query, row in zip(queries[start : start + self.chunk], scores, strict=True):
                run[query["query_id"]] = self.top_hits(row)
        qrels = {q["query_id"]: self.qrels[q["query_id"]] for q in queries}
        return score_run(run, qrels, ["mrr@10"])["mrr@10"]

    def top_hits(self, scores: np.ndarray) -> dict[str, float]:
        """Same order as VectorRetriever.search, without sorting all candidates."""
        kth = np.partition(scores, -self.top_k)[-self.top_k] if len(scores) > self.top_k else scores.min()
        keep = np.flatnonzero(scores >= kth)  # keeps every recipe tied with the k-th one
        hits = rank_top_k(self.ids[keep], scores[keep], self.top_k)
        return {hit.recipe_id: float(len(hits) - rank) for rank, hit in enumerate(hits)}


def load_cv_data(directory: Path) -> tuple[list[str], list[str], list[list[str]], list[dict], dict]:
    """Outer-train recipes (ids, groups, normalised docs) and the normalised train queries + qrels."""
    train_ids = set(read_json(directory / "splits.json")["train"])
    recipes = sorted(
        (r for r in read_jsonl(directory / "corpus.jsonl") if r["recipe_id"] in train_ids),
        key=lambda r: r["recipe_id"],
    )
    queries, qrels = load_split(directory, "train")
    queries = [{**q, "ingredients": normalize_ingredients(q["ingredients"])} for q in queries]
    docs = normalize_all([r["ingredients_normalized"] for r in recipes])
    return [r["recipe_id"] for r in recipes], [r["group_id"] for r in recipes], docs, queries, qrels


def lsa_estimator() -> Pipeline:
    """The LSA baseline without its normalise step (docs are normalised up front here)."""
    return Pipeline(build_lsa_pipeline(n_components=LSA_COMPONENTS).steps[1:])


def run_cv(
    ids, groups, docs, queries, qrels, param_grid=PARAM_GRID, n_folds=N_FOLDS, estimator=None
) -> GridSearchCV:
    search = GridSearchCV(
        estimator if estimator is not None else TfidfVectorizer(analyzer=identity),
        param_grid,
        scoring=RetrievalScorer(ids, docs, queries, qrels),
        cv=GroupKFold(n_splits=n_folds),
        refit=False,  # the E02 bundle is fitted on the whole outer train by src.baseline --tuned
        return_train_score=True,
        error_score="raise",
    )
    return search.fit(docs, ids, groups=groups)


def results_frame(search: GridSearchCV) -> pd.DataFrame:
    frame = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(list(frame["params"]), dtype=object)  # object keeps norm=None (not NaN)
    if "analyzer" in params:
        params["tokens"] = params.pop("analyzer").map(TOKENS)
    keep = ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "mean_fit_time"]
    out = pd.concat([params, frame[keep], frame["rank_test_score"]], axis=1)
    names = {"test_score": "cv_mrr@10", "train_score": "train_mrr@10"}
    return out.rename(columns=lambda c: next((c.replace(k, v) for k, v in names.items() if k in c), c))


def summary_row(search: GridSearchCV, frame: pd.DataFrame, params: dict) -> dict:
    """The results row of one grid point, as JSON-ready Python values."""
    row = frame.loc[search.cv_results_["params"].index(params)].to_dict()
    return {k: v.item() if hasattr(v, "item") else v for k, v in row.items()}


def config_params(row: dict) -> dict:
    """The values that go into configs/default.yaml (baselines.tfidf_tuned)."""
    return {k: row[k] for k in ("tokens", "min_df", "sublinear_tf", "norm")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model", default="tfidf", choices=["tfidf", "lsa"], help="lsa: CV score of the untuned LSA only"
    )
    parser.add_argument("--out", default="reports", help="folder for <model>_cv5.csv / <model>_cv5.json")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    ids, groups, docs, queries, qrels = load_cv_data(directory)
    tfidf = args.model == "tfidf"
    started = time.perf_counter()
    if tfidf:
        search = run_cv(ids, groups, docs, queries, qrels)
    else:
        search = run_cv(ids, groups, docs, queries, qrels, param_grid={}, estimator=lsa_estimator())
    cv_seconds = time.perf_counter() - started

    frame = results_frame(search)
    best = summary_row(search, frame, search.best_params_)
    rows = {"default": best}
    if tfidf:
        rows = {"best": best, "default_E01": summary_row(search, frame, E01_PARAMS)}
    frame = frame.sort_values("rank_cv_mrr@10")
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    name = f"{args.model}_cv5"
    frame.to_csv(out / f"{name}.csv", index=False)
    summary = {
        "model": args.model,
        "profile": cfg.data.profile,
        "corpus_sha256": sha256_file(directory / "corpus.jsonl"),
        "n_folds": N_FOLDS,
        "n_recipes": len(ids),
        "n_groups": len(set(groups)),
        "n_queries": len(queries),
        "n_candidates": len(frame),
        "cv_seconds": round(cv_seconds, 1),
        **rows,
        **({"tfidf_tuned": config_params(best)} if tfidf else {"n_components": LSA_COMPONENTS}),
    }
    write_json(out / f"{name}.json", summary)

    print(frame.to_string(index=False))
    for label, row in rows.items():
        print(
            f"{label}: cv MRR@10 {row['mean_cv_mrr@10']:.4f} ± {row['std_cv_mrr@10']:.4f} | "
            f"train proxy {row['mean_train_mrr@10']:.4f}" + (f" | {config_params(row)}" if tfidf else "")
        )
    print(f"{len(frame)} candidates x {N_FOLDS} folds in {cv_seconds:.0f} s -> {out / f'{name}.json'}")

    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        run_name = f"E02-tfidf-cv5-{cfg.data.profile}" if tfidf else f"lsa-cv5-{cfg.data.profile}"
        with mlflow.start_run(run_name=run_name) as run:
            params = {
                "model": args.model,
                "profile": cfg.data.profile,
                "cv": f"GroupKFold({N_FOLDS}) on outer train, groups=group_id",
                "scorer": "mrr@10 (ranx) on train queries of the held-out fold",
                "corpus_sha256": summary["corpus_sha256"][:12],
                "config_sha256": fingerprint(cfg)[:12],
            }
            metrics = {
                "cv_mrr_at_10_mean": best["mean_cv_mrr@10"],
                "cv_mrr_at_10_std": best["std_cv_mrr@10"],
                "train_proxy_mrr_at_10": best["mean_train_mrr@10"],
                "cv_seconds": cv_seconds,
            }
            if tfidf:
                default = rows["default_E01"]
                params.update(
                    experiment="E02",
                    grid=str({k: [TOKENS.get(v, v) for v in vs] for k, vs in PARAM_GRID.items()}),
                    **{f"best_{k}": v for k, v in config_params(best).items()},
                )
                metrics.update(
                    e01_cv_mrr_at_10_mean=default["mean_cv_mrr@10"],
                    e01_cv_mrr_at_10_std=default["std_cv_mrr@10"],
                    e01_train_proxy_mrr_at_10=default["mean_train_mrr@10"],
                )
            else:
                params["n_components"] = LSA_COMPONENTS
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out / f"{name}.csv"))
            mlflow.log_artifact(str(out / f"{name}.json"))
            print(f"mlflow run: {run.info.run_id} ({args.tracking_uri})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
