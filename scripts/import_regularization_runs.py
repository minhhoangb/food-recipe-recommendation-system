"""Import the actual Colab MLflow export, preserving metric timestamps and source IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from src.utils import sha256_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/regularization_ablation/colab_export.json"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args(argv)
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["hardware"]["colab"] is True
    assert len(summary["arms"]) == 2
    assert {arm["weight_decay"] for arm in summary["arms"]} == {0.0, 0.01}
    assert len({arm["initial_weights_sha256"] for arm in summary["arms"]}) == 1
    client = MlflowClient(args.tracking_uri)
    experiment = client.get_experiment_by_name("recipe-regularization-ablation")
    eid = experiment.experiment_id if experiment else client.create_experiment("recipe-regularization-ablation")
    mapping = {}
    for record in payload["mlflow_export"]:
        original_id = record["info"]["run_id"]
        assert record["info"]["status"] == "FINISHED"
        found = client.search_runs([eid], filter_string=f"tags.source_colab_run_id = '{original_id}'")
        if found:
            mapping[original_id] = found[0].info.run_id
            continue
        tags = {**record["tags"], "source_colab_run_id": original_id,
                "source_export_sha256": sha256_file(args.source), "imported_from": "colab",
                "colab_notebook": "https://colab.research.google.com/drive/178UsXUi0OhmEOM7PgDbyIY4UGHYX9L4M"}
        if "mlflow.parentRunId" in tags:
            tags["mlflow.parentRunId"] = mapping[tags["mlflow.parentRunId"]]
        run = client.create_run(eid, start_time=record["info"]["start_time"], tags=tags)
        rid = run.info.run_id
        mapping[original_id] = rid
        client.log_batch(rid, params=[Param(k, v) for k, v in record["params"].items()])
        metrics = [Metric(key, m["value"], m["timestamp"], m["step"])
                   for key, history in record["history"].items() for m in history]
        for start in range(0, len(metrics), 500):
            client.log_batch(rid, metrics=metrics[start:start + 500])
        client.log_artifact(rid, str(args.source))
        client.set_terminated(rid, status="FINISHED", end_time=record["info"]["end_time"])
    summary["local_run_mapping"] = mapping
    write_json(args.source.parent / "summary.json", summary)
    print(json.dumps({"experiment_id": eid, "run_mapping": mapping}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
