"""Join frozen E10 results to existing, run-backed Train/Validation/CV evidence. No retrieval."""

from __future__ import annotations

import math

import mlflow
import pandas as pd

from src.utils import REPO_ROOT, read_json, sha256_file, write_json


def build() -> pd.DataFrame:
    reports = REPO_ROOT / "reports"
    out = reports / "final"
    state = read_json(out / "e10_execution.json")
    if state["status"] != "complete":
        raise ValueError("E10 has not completed; do not create a final results table")
    frozen = read_json(out / "frozen_config.json")
    if sha256_file(out / "frozen_config.json") != state["freeze_sha256"]:
        raise ValueError("Freeze manifest changed after E10")
    history = pd.read_csv(reports / "comparison.csv", keep_default_na=False)
    test = pd.read_csv(out / "comparison_test.csv")
    frame = history.merge(test, on="experiment", validate="one_to_one")
    if len(frame) != len(frozen["systems"]):
        raise ValueError("Master table is missing a frozen system")
    provenance = read_json(reports / "finetune_cpu1ep_provenance.json")
    client = mlflow.MlflowClient("sqlite:///mlflow.db")
    # Check source IDs and values rather than copying unchecked historical claims.
    for row in frame.to_dict("records"):
        for prefix, run_column in (
            ("train", "mlflow_train_run"),
            ("val", "mlflow_val_run"),
            ("test", "mlflow_test_run"),
        ):
            run = client.get_run(row[run_column])
            for metric in ("recall@5", "recall@10", "mrr@10"):
                observed = run.data.metrics[metric.replace("@", "_at_")]
                if abs(observed - float(row[f"{prefix}_{metric}"])) > 1e-10:
                    raise ValueError(
                        f"Stored metric disagrees with run: {row['experiment']}/{prefix}/{metric}"
                    )
    frame["training_seconds"] = [
        provenance["observed_training"]["train_seconds"]
        if r.experiment == "FT"
        else (0.0 if r.model == "minilm_pretrained" else float(r.fit_seconds))
        for r in frame.itertuples()
    ]
    frame["training_time_note"] = [
        "historical measured CPU fine-tuning; corpus encoding is separate"
        if label == "FT"
        else (
            "pretrained weights; no training on this dataset"
            if label in ("E04", "E05")
            else "historical fit/index + corpus transform; hardware differs from current serving benchmark"
        )
        for label in frame.experiment
    ]
    frame["training_time_run"] = [
        provenance["mlflow_run_id"] if r.experiment == "FT" else r.mlflow_train_run
        for r in frame.itertuples()
    ]
    frame["cv_run"] = frame.experiment.map(
        {
            "E01": "dac82cb580e8403082cd9d0fabb40448",
            "E02": "dac82cb580e8403082cd9d0fabb40448",
            "LSA": "d46efe09f7dd4835841f66b860d4d433",
        }
    ).fillna("")
    frame["deployment_marker"] = [
        "*" if label == frozen["selected_deployment"] else "" for label in frame.experiment
    ]
    for row in frame.to_dict("records"):
        if row["cv_run"]:
            source = client.get_run(row["cv_run"]).data.metrics
            prefix = "e01_" if row["experiment"] == "E01" else ""
            for statistic in ("mean", "std"):
                expected = float(row[f"cv_mrr@10_{statistic}"])
                if abs(source[f"{prefix}cv_mrr_at_10_{statistic}"] - expected) > 1e-10:
                    raise ValueError("CV score disagrees with its MLflow source")
    frame.to_csv(out / "master_comparison.csv", index=False)
    lines = [
        "| Model | Train MRR@10 | Test MRR@10 | CV MRR@10 | Training/index s | Tuned? | Notes |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for row in frame.to_dict("records"):
        cv = (
            f"{float(row['cv_mrr@10_mean']):.4f} ± {float(row['cv_mrr@10_std']):.4f}"
            if row["cv_mrr@10_mean"] != ""
            else "N/A"
        )
        note = ("Selected before test; " if row["deployment_marker"] else "") + (row["cv_note"] or "")
        lines.append(
            f"| {row['experiment']}{row['deployment_marker']} {row['model']} | "
            f"{row['train_mrr@10']:.4f} | {row['test_mrr@10']:.4f} | {cv} | "
            f"{row['training_seconds']:.3f} | {row['tuned']} | {note} |"
        )
    text = "\n".join(lines) + "\n\n* = selected for deployment using validation before E10. "
    text += (
        "Train and test are origin-recipe retrieval proxies, not human relevance. "
        "Train/Validation/Test, training-time and CV run IDs are in `master_comparison.csv`. "
        "Training/index/encoding times have different meanings and hardware; do not rank their speed.\n"
    )
    (out / "master_comparison.md").write_text(text, encoding="utf-8")
    readme = REPO_ROOT / "README.md"
    contents = readme.read_text(encoding="utf-8")
    start = contents.index("## Experiment results (Master Comparison Table)")
    end = contents.index("## API usage", start)
    contents = (
        contents[:start]
        + "## Experiment results (Master Comparison Table)\n\n"
        + text
        + (
            "\nFrozen protocol and one-run ledger: [reports/final](reports/final). "
            "Final human labels and agreement remain pending; see [the judging instructions]"
            "(docs/final_annotation_workflow.md).\n\n"
        )
        + contents[end:]
    )
    readme.write_text(contents, encoding="utf-8")
    write_json(
        out / "master_provenance.json",
        {
            "e10_parent_run": state["mlflow_parent_run"],
            "freeze_sha256": state["freeze_sha256"],
            "historical_comparison_sha256": sha256_file(reports / "comparison.csv"),
            "ft_training_run": provenance["mlflow_run_id"],
            "test_metrics_verified_against_mlflow": True,
            "selection_changed_after_test": False,
        },
    )

    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        return value

    write_json(
        out / "significance_test_export.json",
        {
            "source_sha256": sha256_file(out / "significance_test.json"),
            "undefined_p_value": "null means undefined (e.g. identical paired scores); no win is inferred",
            "report": finite_json(read_json(out / "significance_test.json")),
        },
    )
    return frame


if __name__ == "__main__":
    print(build()[["experiment", "train_mrr@10", "val_mrr@10", "test_mrr@10", "deployment_marker"]])
