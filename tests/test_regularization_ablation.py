"""Guard the actual regularization knob and the fixed-batch loss protocol."""
from unittest.mock import MagicMock

import pytest
import torch

from scripts.run_regularization_ablation import batches, measured_loss
from src.finetune import finetune_model


def test_weight_decay_is_forwarded_as_fit_argument(monkeypatch, tmp_path):
    model = MagicMock()
    monkeypatch.setattr("src.finetune.SentenceTransformer", lambda *a, **k: model)
    monkeypatch.setattr("src.finetune.losses.MultipleNegativesRankingLoss", lambda m: object())
    _, summary = finetune_model([{"query": "a", "positive": "a b"}], weight_decay=0.0,
                                 device="cpu", output_path=tmp_path)
    assert model.fit.call_args.kwargs["weight_decay"] == 0.0
    assert "weight_decay" not in model.fit.call_args.kwargs["optimizer_params"]
    assert summary["weight_decay"] == 0.0


def test_full_batches_have_no_duplicate_groups_and_are_reproducible():
    pairs = [{"recipe_id": str(i), "group_id": str(i // 2)} for i in range(16)]
    schedule = batches(pairs, 4, 42)
    assert schedule == batches(pairs, 4, 42)
    assert schedule
    for batch in schedule:
        assert len(batch) == len({pairs[i]["group_id"] for i in batch}) == 4


def test_loss_probes_use_eval_mode_and_disable_gradients():
    class Model:
        training = True

        def eval(self):
            self.training = False

        def tokenize(self, texts):
            return {"input_ids": torch.tensor([[len(t)] for t in texts])}

    model = Model()

    def loss(features, labels):
        assert not model.training
        assert not torch.is_grad_enabled()
        return torch.tensor(0.75)

    pairs = [{"query": "a", "positive": "b"}] * 4
    assert measured_loss(model, loss, pairs, [[0, 1], [2, 3]], "cpu") == 0.75
    with pytest.raises(ValueError, match="full batches"):
        measured_loss(model, loss, pairs, [], "cpu")
