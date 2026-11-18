"""Toy-batch unit tests for the phase-5 trainer (src/wtss/experiments/phase5.py). CPU only, untrained weights, tiny images.

    nice -n 19 env OMP_NUM_THREADS=2 python -m pytest -q tests/test_phase5.py
"""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

torch.set_num_threads(2)
from wtss.experiments import phase5 as P5  # noqa: E402

S = 64


def _toy(n=8, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"t{i}" for i in range(n)]
    y = np.array([i % 2 for i in range(n)]); a = np.array([(i // 2) % 2 for i in range(n)])
    imgs, rois = {}, {}
    for i, k in enumerate(ids):
        arr = (rng.random((S, S, 3)) * 60).astype(np.uint8)
        r = np.zeros((S, S), np.uint8); r[10:40, 5:30] = 1
        arr[r > 0] = 250                                   # ROI pixels bright, so alignment can be checked
        imgs[k], rois[k] = arr, r
    df = pd.DataFrame({"image_id": ids, "y": y, "a": a})
    render = {"erm": lambda i: Image.fromarray(imgs[i]), "mask": lambda i: Image.fromarray(imgs[i] * rois[i][..., None])}
    return df, render, (lambda i: rois[i])


def test_joint_augment_keeps_roi_aligned():
    df, render, roi = _toy()
    for ep in range(6):
        ds = P5.P5Images(df, render, "erm", size=S, train=True, norm="unit", roi_fn=roi, seed=1); ds.epoch = ep
        for i in range(len(ds)):
            x, m, _, _ = ds[i]
            assert m.shape == (1, S, S) and set(np.unique(m.numpy())) <= {0.0, 1.0}
            assert (x[:, m[0] > 0] > 0.95).all()          # every ROI pixel is still a bright pixel after the joint transform


def test_dataset_is_reproducible_and_normalised():
    df, render, roi = _toy()
    a = P5.P5Images(df, render, "erm", size=S, train=True, seed=3); b = P5.P5Images(df, render, "erm", size=S, train=True, seed=3)
    assert torch.equal(a[2][0], b[2][0])                    # same (seed, epoch, index) -> same draw
    x = P5.P5Images(df, render, "erm", size=S, train=False)[0][0]
    assert x.dtype == torch.float32 and x.min() < 0          # ImageNet normalisation applied


def test_early_stopping_patience_and_best():
    es = P5.EarlyStopping(patience=5, max_epochs=48)
    seq = [0.5, 0.6, 0.55, 0.58, 0.59, 0.6, 0.55]
    stops = [es.step(e, v)[1] for e, v in enumerate(seq)]
    assert es.best_epoch == 1 and es.best == 0.6 and stops[-1] and not any(stops[:-1])   # ties do not count as improvement
    es2 = P5.EarlyStopping(patience=5, max_epochs=3)
    assert [es2.step(e, 0.5 + e)[1] for e in range(3)] == [False, False, True]
    es3 = P5.EarlyStopping(5, 48)
    assert es3.step(0, float("nan")) == (False, False)      # NaN never counts as best


def test_rrclarc_aux_matches_manual_and_vanishes_orthogonal():
    torch.manual_seed(0)
    g = torch.randn(3, 4, 5, 5); cav = torch.nn.functional.normalize(torch.randn(1, 4), dim=1)
    gm = g.mean((2, 3))                                      # l2_mean: the spatially averaged gradient, one row per position
    manual = ((gm @ cav.T) ** 2).mean()
    assert torch.allclose(P5.rrclarc_aux_loss(g, cav), manual, atol=1e-6)
    q, _ = torch.linalg.qr(torch.cat([cav.T, torch.randn(4, 3)], 1))
    g_orth = (q[:, 1:] @ torch.randn(3, 3 * 25)).T.reshape(3, 5, 5, 4).permute(0, 3, 1, 2)
    assert P5.rrclarc_aux_loss(g_orth, cav) < 1e-10


def test_rrclarc_criterion_random_signs():
    torch.manual_seed(0)
    y = torch.ones(4, 2)
    v = P5.rrclarc_criterion(y)
    assert v.shape == (4,) and set(v.tolist()) <= {-2.0, 0.0, 2.0}


def test_pahde_cav_signal_points_to_the_artifact_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)); a = rng.integers(0, 2, 200); X[:, 3] += 3 * a
    cav = P5.compute_rrclarc_cav(torch.tensor(X, dtype=torch.float32), a, "signal")
    assert cav.shape == (1, 6) and int(cav.abs().argmax()) == 3


def test_resnet50d_hooks_and_rrclarc_step():
    torch.manual_seed(0)
    m = P5.make_resnet50d(pretrained=False)
    names = dict(m.named_modules())
    assert all(n in names for n in ("identity_0", "identity_1", "identity_2", "last_conv"))
    x = torch.randn(2, 3, S, S); y = torch.tensor([0, 1])
    acts = []
    h = names["identity_2"].register_forward_hook(lambda mm, i, o: acts.append(o)); m(x); h.remove()
    cav = torch.nn.functional.normalize(torch.randn(1, acts[0].shape[1]), dim=1)
    rr = P5.RRClarcStep(m, "identity_2", cav, lam=1e3)
    loss, ce, aux = rr.loss(x, y)
    assert torch.isfinite(loss) and loss.dtype == torch.float32 and aux >= 0
    loss.backward()
    assert m.layer1[0].conv1.weight.grad is not None        # all parameters train (A18 finding)


def test_isnet_training_step_on_toy_batch():
    torch.manual_seed(0)
    net = P5.make_isnet(pretrained=False)
    x = torch.rand(2, 3, S, S); m = torch.zeros(2, 1, S, S); m[:, :, 16:48, 16:48] = 1
    loss = net.training_step((x, m, torch.tensor([[0], [1]])), 0)
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    loss.backward()
    df, render, roi = _toy(4)
    ds = P5.P5Images(df, render, "erm", size=S, norm="unit", roi_fn=roi)
    p = P5.predict("isnet", net, P5.loader(ds, 2, False, 0), "cpu")
    assert p.shape == (4,) and np.all((p >= 0) & (p <= 1)) and net.heat is True


def test_effb0_features_and_fit_fp32_with_resume(tmp_path):
    torch.manual_seed(0)
    m = P5.make_effb0(pretrained=False)
    assert P5.effb0_features(m, torch.randn(2, 3, S, S)).shape == (2, 1280)
    df, render, roi = _toy(8)
    tr = P5.P5Images(df, render, "mask", size=S, train=True); va = P5.P5Images(df, render, "mask", size=S)
    cfg = P5.TrainCfg(epochs=2, patience=5, bs=4, workers=0, lr=1e-4)
    r = P5.fit("armd", m, tr, va, cfg, tmp_path, "cpu")
    log = pd.read_csv(tmp_path / "log.csv")
    assert len(log) == 2 and (tmp_path / "best.pt").exists() and r["epochs_run"] == 2
    r2 = P5.fit("armd", m, tr, va, P5.TrainCfg(epochs=3, patience=5, bs=4, workers=0), tmp_path, "cpu")   # resumes
    assert len(pd.read_csv(tmp_path / "log.csv")) == 3 and r2["epochs_run"] == 3


def test_armd_head_removes_the_marker_direction_and_uses_the_grid():
    rng = np.random.default_rng(0)
    n = 4000                                                 # enough minority images for a precise within-class difference
    y = rng.integers(0, 2, n)
    a_tr = np.where(rng.random(n) < 0.9, y, 1 - y)          # marker 90/10 with the label in training
    sig = rng.normal(size=n) + 0.8 * y
    X = np.c_[sig, 3.0 * a_tr, rng.normal(size=(n, 4))]
    yv = rng.integers(0, 2, n); av = rng.integers(0, 2, n)   # clean validation: marker independent of the label
    Xv = np.c_[rng.normal(size=n) + 0.8 * yv, 3.0 * av, rng.normal(size=(n, 4))]
    f, ch = P5.armd_head(X, y, a_tr, Xv, yv)
    assert ch["C"] in P5.C_GRID and ch["mode"] in ("pooled", "per_class") and ch["val_clean_auroc"] > 0.65
    # flipping only the marker must not move the score much (its direction was projected out)
    Xf = Xv.copy(); Xf[:, 1] = 3.0 * (1 - av)
    assert np.abs(f(Xf) - f(Xv)).mean() < 0.1 * np.abs(f(Xv)).mean() + 1e-6
