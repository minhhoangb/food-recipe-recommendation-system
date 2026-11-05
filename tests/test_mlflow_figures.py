"""Missing measurements must fail instead of producing invented curves."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import generate_mlflow_figures


def test_missing_loss_history_cannot_generate_a_figure(monkeypatch, tmp_path):
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(tags={"controlled_ablation": "true"}, params={"weight_decay": "0.0"}),
    )
    client.get_metric_history.return_value = []
    monkeypatch.setattr(generate_mlflow_figures, "MlflowClient", lambda uri: client)
    output = tmp_path / "missing.png"
    with pytest.raises(ValueError, match="missing measured train_loss"):
        generate_mlflow_figures.main(["--run-ids", "a", "b", "--out", str(output)])
    assert not output.exists()
