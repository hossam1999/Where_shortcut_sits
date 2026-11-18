"""Toy-batch tests of the arm-d trainer (A20): early stopping, one training run, checkpoint/resume, feature shapes. CPU."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("armd", Path(__file__).resolve().parents[1] / "scripts/replication/armd_train.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def test_early_stop():
    es = A.EarlyStop(patience=2)
    assert es.step(0.5) and not es.stop
    assert not es.step(0.5) and not es.stop       # equal is not an improvement
    assert not es.step(0.4) and es.stop
    assert es.best == 0.5


class Toy:
    n_out, size = 3, 32
    sets = ("train", "val", "test")
    train_set, val_sets = "train", ("val",)

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.y = {k: torch.randint(0, 3, (40,), generator=g) for k in self.sets}
        self.x = {k: torch.randn(40, 3, 32, 32, generator=g) + self.y[k][:, None, None, None].float() for k in self.sets}

    def dataset(self, k, train):
        return torch.utils.data.TensorDataset(self.x[k], self.y[k])

    def val_metric(self, preds):
        y, p = preds["val"]
        return float((p.argmax(1) == y).mean())

    def labels(self, k):
        return self.y[k].numpy(), np.zeros(40, int)


def test_train_resume_features(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "BS", 8)
    S = Toy()
    A.train(S, tmp_path, workers=0, max_batches=2, max_epochs=2, pretrained=False)
    log = (tmp_path / "log.csv").read_text().splitlines()
    assert len(log) == 1 + 2 and (tmp_path / "best.pt").exists() and (tmp_path / "checkpoint.pt").exists()
    A.train(S, tmp_path, workers=0, max_batches=2, max_epochs=3, pretrained=False)     # resumes after epoch 1
    c = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert c["epoch"] == 2 and len(c["log"]) == 3
    A.features(S, tmp_path, workers=0, pretrained=False)
    F = np.load(tmp_path / "features.npz")
    assert F["train"].shape == (40, 1280) and F["test_logits"].shape == (40, 3)
