"""Stricter leakage groups: pHash/frame-block groups united with DINOv2 near-duplicate components
(docs/PREREGISTRATION_EMBEDDING_GROUPS.md).
  python scripts/data/embedding_groups.py   # -> <cohort dir>/groups_emb.csv, results/leakage/embedding_groups.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from wtss import paths

COHORTS = {"thyroid": ("us/tncd/thyroid_cohort.csv", "group_ph8", "us/tncd"),
           "ovary": ("ovary/ovary_cohort.csv", "group", "ovary"),
           "capsule": ("capsule/capsule_cohort.csv", "group", "capsule")}


def components(n, pairs):
    if len(pairs) == 0:
        return np.arange(n)
    i, j = np.asarray(pairs).T
    return connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[1]


def main():
    summary = {}
    for co, (csv, gcol, d) in COHORTS.items():
        c = pd.read_csv(paths.DATA / csv)
        z = np.load(paths.CACHE / "features" / co / "dinov2_b14_518" / "erm.npz", allow_pickle=True)
        X = z["X"].astype(np.float32); X -= X.mean(0)  # amendment: centre first (raw cosine ~0.99 for all ovary pairs)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        pos = {k: i for i, k in enumerate(c.image_id.astype(str))}
        idx = np.array([pos[str(k)] for k in z["ids"]])
        G = X @ X.T
        np.fill_diagonal(G, -1)
        old = pd.factorize(c[gcol].astype(str))[0]
        old_pairs = []  # chain members of each existing group
        for g in np.unique(old):
            m = np.flatnonzero(old == g)
            old_pairs += list(zip(m[:-1], m[1:]))
        chosen = None
        for tau in np.round(np.arange(0.99, 0.795, -0.01), 2):
            a, b = np.nonzero(np.triu(G >= tau, 1))
            lab = components(len(c), old_pairs + list(zip(idx[a], idx[b])))
            big = np.bincount(lab).max() / len(c)
            if big <= 0.05:
                chosen = (tau, lab, big, len(a))
            else:
                break
        tau, lab, big, ne = chosen
        c["group_emb"] = [f"e{x}" for x in lab]
        c[["image_id", "group_emb"]].to_csv(paths.DATA / d / "groups_emb.csv", index=False)
        summary[co] = {"tau": float(tau), "n_images": len(c), "groups_old": int(len(np.unique(old))),
                       "groups_emb": int(len(np.unique(lab))), "largest_group_share": float(big), "embedding_edges": int(ne)}
        print(co, summary[co], flush=True)
    out = paths.ensure(paths.RESULTS / "leakage")
    json.dump(summary, open(out / "embedding_groups.json", "w"), indent=1)


if __name__ == "__main__":
    main()
