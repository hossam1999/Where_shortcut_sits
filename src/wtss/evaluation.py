"""Evaluation: frozen clean-val threshold, AUROC, worst-group accuracy, same-head |Δp|."""
from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .stats import safe_auc


def select_threshold_clean_val(y, prob) -> Tuple[float, float]:
    """Threshold chosen only on clean validation by balanced accuracy, then frozen."""
    cands = np.unique(np.r_[0.0, prob, 1.0])
    if len(cands) > 2000:
        cands = np.quantile(cands, np.linspace(0, 1, 2000))
    best_t, best = 0.5, -1.0
    for t in cands:
        s = balanced_accuracy_score(y, (prob >= t).astype(int))
        if s > best:
            best, best_t = float(s), float(t)
    return best_t, best


def group_accuracy(y, a, pred):
    details, vals = {}, []
    for yy in (0, 1):
        for aa in (0, 1):
            m = (y == yy) & (a == aa)
            if m.sum() == 0:
                details[f"y{yy}_a{aa}"] = {"n": 0, "accuracy": float("nan")}
                continue
            acc = float((pred[m] == y[m]).mean())
            details[f"y{yy}_a{aa}"] = {"n": int(m.sum()), "accuracy": acc}
            vals.append(acc)
    return (float(min(vals)) if len(vals) == 4 else float("nan")), details


def evaluate(clf, threshold, X, y, ids, a, meta: Dict) -> Tuple[Dict, pd.DataFrame]:
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    wga, groups = group_accuracy(y, a, pred)
    row = {**meta, "auc": safe_auc(y, prob),
           "auc_artifact_present": safe_auc(y[a == 1], prob[a == 1]) if (a == 1).any() else float("nan"),
           "auc_artifact_absent": safe_auc(y[a == 0], prob[a == 0]) if (a == 0).any() else float("nan"),
           "n": int(len(y)), "n_pos": int(y.sum()), "n_artifact_present": int((a == 1).sum()),
           "accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "worst_group_accuracy": wga, "group_details_json": json.dumps(groups),
           "threshold_clean_val": float(threshold)}
    pdf = pd.DataFrame({**{k: [v] * len(y) for k, v in meta.items()}, "image_id": np.asarray(ids).astype(str),
                        "y": np.asarray(y).astype(int), "artifact_present": np.asarray(a).astype(int),
                        "prob": prob.astype(float), "pred": pred})
    return row, pdf


def same_head_counterfactual(preds: pd.DataFrame, env: str = "test_rev",
                             keys=("method", "overlap", "seed")) -> pd.DataFrame:
    """|p(artifact version) - p(clean version)| for artifact-present images, same trained head."""
    per = []
    for k, q in preds.groupby(list(keys)):
        c = q[q.env == "clean"][["image_id", "y", "prob"]].rename(columns={"prob": "p_clean"})
        art = q[(q.env == env) & (q.artifact_present == 1)][["image_id", "y", "prob"]].rename(columns={"prob": "p_artifact"})
        z = art.merge(c, on=["image_id", "y"])
        if z.empty:
            continue
        z["abs_delta_p"] = (z.p_artifact - z.p_clean).abs()
        for name, v in zip(keys, k if isinstance(k, tuple) else (k,)):
            z[name] = v
        per.append(z)
    return pd.concat(per, ignore_index=True) if per else pd.DataFrame()
