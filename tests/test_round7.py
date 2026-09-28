"""Round 7: thresholds matched on the test set (docs/PREREGISTRATION_ROUND7.md). Synthetic predictions only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve().parents[1] / "scripts" / "round7" / "matched_thresholds.py"
_spec = importlib.util.spec_from_file_location("round7_matched", _p)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def _preds(shift_conflict: float, seeds=(1, 2), n=400):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    base = rng.normal(0, 1, n) + 1.2 * y
    rows = []
    for s in seeds:
        noise = rng.normal(0, 0.1, n)
        conf = (y == 1) & (a == 1)
        for m, sc in (("erm", base + noise), ("mask", base + noise - shift_conflict * conf)):
            rows.append(pd.DataFrame({"image_id": [f"i{j}" for j in range(n)], "y": y, "artifact_present": a,
                                      "prob": 1 / (1 + np.exp(-sc)), "method": m, "env": "clean", "seed": s}))
    return pd.concat(rows, ignore_index=True)


def _row(rows, match, metric):
    return next(r for r in rows if r["match"] == match and r["metric"] == metric)


def test_identical_arms_give_zero():
    rows = M.analyse(("t", "c", _preds(0.0), 1, 50))
    for m in M.MATCH:
        r = _row(rows, m, "sens_conflict")
        assert abs(r["delta"]) < 1e-12 and abs(r["ci95_lo"]) < 1e-12 and abs(r["ci95_hi"]) < 1e-12


def test_lower_conflict_scores_lower_matched_sensitivity():
    rows = M.analyse(("t", "c", _preds(1.5), 1, 100))
    r = _row(rows, "M-spec80", "sens_conflict")
    assert r["delta"] < -0.2 and r["ci95_hi"] < 0
    spec = _row(rows, "M-spec80", "spec")  # both arms at the same test specificity
    assert abs(spec["delta"]) < 0.02 and spec["erm_value"] >= 0.80 and spec["mask_value"] >= 0.80


def test_holm():
    assert np.allclose(M.holm(np.array([0.01, 0.04])), [0.02, 0.04])
    assert np.allclose(M.holm(np.array([0.04, 0.01])), [0.04, 0.02])
