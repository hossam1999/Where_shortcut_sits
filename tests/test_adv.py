"""Toy tests of the exploratory candidate (b) trainer: gradient reversal, one training run with resume, features. CPU."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "newmethod"), str(ROOT / "scripts" / "replication"), str(ROOT / "tests")]
spec = importlib.util.spec_from_file_location("adv", ROOT / "scripts/newmethod/adv_train.py")
ADV = importlib.util.module_from_spec(spec); spec.loader.exec_module(ADV)
from test_armd import Toy  # noqa: E402


def test_grl_reverses_and_scales():
    x = torch.ones(3, requires_grad=True)
    ADV.GRL.apply(x, 2.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.5))


class ToyTube(Toy):
    val_sets = ("val",)

    def labels(self, k):
        return self.y[k].numpy(), (np.arange(40) % 2)


def test_adv_train_resume_features(tmp_path, monkeypatch):
    monkeypatch.setattr(ADV.A, "BS", 8)
    S = ToyTube()
    S.train_set = "train"
    ADV.train(S, tmp_path, alpha=1.0, workers=0, max_batches=2, max_epochs=2, pretrained=False)
    c = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert c["epoch"] == 1 and "adv" in c and len(c["log"]) == 2
    ADV.train(S, tmp_path, alpha=1.0, workers=0, max_batches=2, max_epochs=3, pretrained=False)
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=False)["epoch"] == 2
