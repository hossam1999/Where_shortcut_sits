import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from wtss import replication_stats as R
from wtss import stats_crossed as SC


def _seeds(n_seeds=3, n=300, K=None, shift=0.5, shared=True, rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    for s in range(n_seeds):
        ids = [f"img{i}" if shared else f"s{s}_img{i}" for i in range(n)]
        if K is None:
            y = rng.integers(0, 2, n)
            a = y + rng.normal(0, 1, n)
            b = y * shift + rng.normal(0, 1, n)
        else:
            y = rng.integers(0, K, n)
            a = np.eye(K)[y] * 1.5 + rng.normal(0, 1, (n, K))
            b = np.eye(K)[y] * 0.8 + rng.normal(0, 1, (n, K))
        out[s] = (ids, y, a, b)
    return out


def test_auroc_replicates_identical_to_stats_crossed():
    seeds = _seeds()
    res = R.paired_bootstrap(seeds, "auroc", n_boot=600, seed=7)
    terms = []
    for s, (ids, y, a, b) in seeds.items():
        terms += [(s, 1.0, ids, y, a), (s, -1.0, ids, y, b)]
    point, arr = SC.replicates(terms, 600, 7)
    assert abs(res["delta"] - np.mean(list(point.values()))) < 1e-12
    lo, hi = np.percentile(arr, [2.5, 97.5])
    assert abs(res["ci95_lo"] - lo) < 1e-12 and abs(res["ci95_hi"] - hi) < 1e-12


def test_point_metrics_match_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 400)
    s = np.eye(3)[y] + rng.normal(0, 1.2, (400, 3))
    for name, ref in (("accuracy", accuracy_score(y, s.argmax(1))),
                      ("macro_f1", f1_score(y, s.argmax(1), average="macro")),
                      ("macro_auroc", roc_auc_score(y, np.exp(s) / np.exp(s).sum(1, keepdims=True),
                                                    multi_class="ovr", average="macro"))):
        f, point = R._metric(name, y, s)
        assert abs(point - ref) < 1e-12
        assert abs(float(f(np.ones((1, len(y))))[0]) - ref) < 1e-12


def test_one_seed_is_image_only_and_detects_a_real_gap():
    seeds = _seeds(n_seeds=1, n=800, K=3)
    for m in ("accuracy", "macro_f1", "macro_auroc"):
        res = R.paired_bootstrap(seeds, m, n_boot=800, seed=1)
        assert res["n_seeds"] == 1 and res["seed_variance_included"] is False
        assert res["delta"] > 0 and res["ci95_lo"] > 0


def test_no_gap_gives_ci_covering_zero():
    rng = np.random.default_rng(5)
    ids = [f"i{k}" for k in range(500)]
    y = rng.integers(0, 2, 500)
    a = y + rng.normal(0, 1, 500)
    res = R.paired_bootstrap({0: (ids, y, a, a.copy())}, "auroc", n_boot=400, seed=2)
    assert res["delta"] == 0 and res["ci95_lo"] <= 0 <= res["ci95_hi"]


def test_holm():
    assert np.allclose(R.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert R.holm([0.5, 0.9]) == [1.0, 1.0]
