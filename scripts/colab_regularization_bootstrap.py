"""Paste this cell into Colab. Downloads only hash-verified public project files."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def run(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


release = "https://github.com/minhhoangb/food-recipe-recommendation-system/releases/download/processed-1f1f4ce505/"
source = Path("/content/ablation-source-v2.zip")
urllib.request.urlretrieve(release + source.name, source)
assert hashlib.sha256(source.read_bytes()).hexdigest() == "3235058b6240324b45a24cc40bfcb23b8dec8eb13e98687a96e0628ace99ac1b"
repo = Path("/content/recipe-regularization")
repo.mkdir(exist_ok=True)
with zipfile.ZipFile(source) as archive:
    archive.extractall(repo)
os.chdir(repo)
run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers==5.1.2",
                "transformers==4.57.6", "mlflow==3.17.2", "omegaconf==2.4.0",
                "ingredient-parser-nlp==2.9.1", "inflect==7.6.0", "pooch==1.9.0"])
run([sys.executable, "scripts/fetch_artifacts.py"])
run([sys.executable, "scripts/verify_data.py"])
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Actual device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "CPU", flush=True)
run([sys.executable, "-m", "scripts.run_regularization_ablation", "--device", device,
                "--train-pairs", "1024", "--val-pairs", "256", "--epochs", "3",
                "--batch-size", "16", "--eval-every", "32"])
summary = json.loads(Path("reports/regularization_ablation/summary.json").read_text())
print(json.dumps(summary, indent=2), flush=True)
from mlflow.tracking import MlflowClient

client = MlflowClient("sqlite:///mlflow.db")
exports = []
for rid in [summary["parent_run_id"]] + [a["run_id"] for a in summary["arms"]]:
    run = client.get_run(rid)
    exports.append({"info": dict(run.info), "params": run.data.params, "tags": run.data.tags,
                    "history": {key: [dict(metric) for metric in client.get_metric_history(rid, key)]
                                for key in run.data.metrics}})
Path("reports/regularization_ablation/mlflow_export.json").write_text(json.dumps(exports, indent=2))
evidence = Path("/content/regularization-evidence")
evidence.mkdir(exist_ok=True)
for path in Path("reports/regularization_ablation").glob("*.json*"):
    shutil.copy2(path, evidence / path.name)
shutil.copy2("mlflow.db", evidence / "mlflow.db")
shutil.make_archive("/content/regularization-colab-evidence", "zip", evidence)
print("ABLATION_COMPLETED", flush=True)
from google.colab import files

files.download("/content/regularization-colab-evidence.zip")
