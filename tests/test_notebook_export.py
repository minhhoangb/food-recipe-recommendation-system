import json
from pathlib import Path

import pytest

from src.utils import REPO_ROOT


def export_cell():
    notebook = json.loads((REPO_ROOT / "notebooks/03_finetune_main_ablation.ipynb").read_text())
    return next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "checkpoint_export_dir = Path(" in "".join(c.get("source", []))
    )


def test_export_refuses_to_publish_pretrained_as_finetuned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def pretrained(*args):
        pytest.fail("export must not load a pretrained fallback")

    with pytest.raises(RuntimeError, match="completed training"):
        exec(export_cell(), {"run_results": {}, "Path": Path, "SentenceTransformer": pretrained})
    assert not (tmp_path / "models").exists()


def test_export_selects_the_measured_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class TrainedModel:
        def __init__(self, weights):
            self.weights = weights

        def save_pretrained(self, directory):
            (Path(directory) / "model.safetensors").write_bytes(self.weights)

    results = {
        "config1_lr2e-5_b32": {"model": TrainedModel(b"lower"), "val_mrr@10": 0.2},
        "config2_lr1e-5_b64": {"model": TrainedModel(b"best"), "val_mrr@10": 0.8},
    }
    exec(export_cell(), {"run_results": results, "Path": Path})
    assert (tmp_path / "models/minilm_finetuned_main/model.safetensors").read_bytes() == b"best"
