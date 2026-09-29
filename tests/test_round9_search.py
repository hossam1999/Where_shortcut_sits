"""Round-9 search: the data firewall and the search heads (docs/ROUND9_SEARCH_LEDGER.md)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"r9t_{name}", ROOT / "scripts" / "round9_search" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FW = _load("firewall")
M = _load("methods")


@pytest.mark.parametrize("seed", [8101, 8202, 8303, 8404, 8505, 9101, 9202, 9303, 9404, 9505, 7])
def test_firewall_rejects_forbidden_and_unknown_seeds(seed):
    with pytest.raises(FW.FirewallError):
        FW.check("trap", "thyroid", [seed])


@pytest.mark.parametrize("kind,cohort", [("trap", "ovary"), ("natural", "isic2020"), ("natural", "ovary")])
def test_firewall_rejects_held_out_and_external_cohorts(kind, cohort):
    with pytest.raises(FW.FirewallError):
        FW.check(kind, cohort, [42])


@pytest.mark.parametrize("env", ["clean", "test_rev", "test"])
def test_firewall_rejects_natural_test_sets(env):
    with pytest.raises(FW.FirewallError):
        FW.check("natural", "thyroid", [42], ["train_corr", env])


def test_firewall_allows_development_data():
    FW.check("trap", "isic", [42, 123, 456, 789, 2026], ["train_corr", "val_groups", "test_rev", "test_corr", "clean"])
    FW.check("natural", "isic_BCN", [42], ["train_corr", "val_groups", "val_eval"])
    FW.check("trap", "thyroid", [99991], smoke=True)
    with pytest.raises(FW.FirewallError):
        FW.check("trap", "thyroid", [99991])  # the smoke seed only in smoke mode


def test_linear_solver_matches_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 5))
    y = (X[:, 0] + rng.normal(0, 1, 400) > 0).astype(int)
    h = M.fit_linear(X, y, C=1.0)
    ref = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000).fit(X, y)
    # same objective up to liblinear/lbfgs intercept handling: coefficients close
    assert np.corrcoef(h.beta, ref.coef_.ravel())[0, 1] > 0.999


def test_backdoor_adjustment_moves_the_artifact_effect_to_the_free_covariate():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 2, n)
    a = (rng.random(n) < np.where(y == 1, 0.9, 0.1)).astype(int)
    X = np.c_[y + rng.normal(0, 1, n), 2.0 * a + rng.normal(0, 0.3, n)]   # feature 1 encodes the artifact only
    plain = M.fit_linear(X, y, C=1.0)
    ba = M.fit_linear(X, y, C=1.0, free=a.astype(float))
    assert abs(ba.beta[1]) < 0.25 * abs(plain.beta[1])                      # the artifact feature is no longer used


def test_group_log_odds_is_the_bias_only_model():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    lo = M.group_log_odds(y, a, w=np.ones(8))
    assert np.allclose(lo[a == 1], np.log(3 / 1), atol=1e-4) and np.allclose(lo[a == 0], np.log(1 / 3), atol=1e-4)
