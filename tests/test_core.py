import numpy as np
import pandas as pd
import pytest

from wtss.methods.insertion import fit_difference_subspace
from wtss.stats import _WeightedAUC, binary_auc, hierarchical_paired_bootstrap, safe_auc
from wtss.synthetic import ENV_RATES, find_best_placements_for_mask, presence_vector


def test_binary_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.normal(size=400) + y, 1)
    assert binary_auc(y, s) == pytest.approx(safe_auc(y, s), abs=1e-12)


def test_weighted_auc_equals_resampled_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)
    w = rng.multinomial(300, np.full(300, 1 / 300), size=5).astype(float)
    got = _WeightedAUC(y, s)(w)
    for k in range(5):
        idx = np.repeat(np.arange(300), w[k].astype(int))
        assert got[k] == pytest.approx(binary_auc(y[idx], s[idx]), abs=1e-12)


def test_fast_and_legacy_bootstrap_agree():
    rng = np.random.default_rng(2)
    rows = []
    for seed in (1, 2, 3):
        y = rng.integers(0, 2, 300)
        pa, pb = y + rng.normal(size=300), 0.5 * y + rng.normal(size=300)
        for m, p in (("a", pa), ("b", pb)):
            rows.append(pd.DataFrame({"seed": seed, "env": "test_rev", "method": m, "image_id": np.arange(300), "y": y, "prob": p}))
    df = pd.concat(rows)
    slow = hierarchical_paired_bootstrap(df, "a", "b", "test_rev", 3000, 0, fast=False)
    fast = hierarchical_paired_bootstrap(df, "a", "b", "test_rev", 3000, 0, fast=True)
    assert fast["seed_delta_mean"] == slow["seed_delta_mean"]
    assert abs(fast["ci95_lo"] - slow["ci95_lo"]) < 0.01 and abs(fast["ci95_hi"] - slow["ci95_hi"]) < 0.01


def test_environment_rates():
    ids = [f"img{i}" for i in range(20000)]
    y = np.r_[np.ones(10000, int), np.zeros(10000, int)]
    for env, (p1, p0) in ENV_RATES.items():
        a = presence_vector(ids, y, 42, env)
        assert a[y == 1].mean() == pytest.approx(p1, abs=0.02)
        assert a[y == 0].mean() == pytest.approx(p0, abs=0.02)


def test_placement_hits_target_overlap():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 1
    pl = find_best_placements_for_mask(m, "x", [0.0, 0.5, 1.0], 20, 6)
    for t in ("0.00", "0.50", "1.00"):
        assert pl[t]["error"] < 0.05


def test_difference_subspace_removes_inserted_direction():
    rng = np.random.default_rng(3)
    X0 = rng.normal(size=(2000, 32))
    U = np.linalg.qr(rng.normal(size=(32, 3)))[0]
    X1 = X0 + (rng.normal(size=(2000, 3)) + 2) @ U.T
    er = fit_difference_subspace(X0, X1, energy=0.95)
    assert er.k == 3
    D = er(X1) - er(X0)
    assert np.abs(D).max() < 1e-4
