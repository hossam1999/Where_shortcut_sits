"""Crossed (cluster x image) bootstrap CIs for the headline claims (docs/PREREGISTRATION_REVIEW3.md, R6).

  python scripts/analysis/crossed_ci.py   -> results/review3/crossed_ci.csv
Every row gives the original hierarchical CI (as reported before) and the crossed CI; missing inputs are skipped.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from wtss import paths
from wtss.stats import crossed_auc_bootstrap, difference_of_deltas, hierarchical_interaction, paired_terms

R = paths.RESULTS
TRAPS = {"ISIC hair": "spec_e13", "Thyroid": "thyroid", "Ovary": "ovary", "Capsule": "capsule"}
TRANS = {"ISIC hair": "isic", "Thyroid": "thyroid", "Ovary": "ovary", "Capsule": "capsule"}
NAT = {"thyroid": "Thyroid (official split)", "isic_BCN": "ISIC, BCN held out", "isic_HAM": "ISIC, HAM held out",
       "isic_MSK": "ISIC, MSK held out", "capsule": "Capsule (held-out frames)"}


def _load(f):
    return pd.read_csv(f) if f.exists() else None


def cross_terms(p):
    return (paired_terms(p[p.trap == "trapB"], "mask", "erm", "test_rev", +1.0) +
            paired_terms(p[p.trap == "trapA"], "mask", "erm", "test_rev", -1.0))


def inter_terms(p, sign=1.0, arm="mask"):
    return (paired_terms(p[np.isclose(p.overlap, 0.0)], arm, "erm", "test_rev", +sign) +
            paired_terms(p[np.isclose(p.overlap, 1.0)], arm, "erm", "test_rev", -sign))


def jobs():
    J = []
    for cohort, base in TRAPS.items():
        for tag in ("repro", "matched", "matched05"):
            p = _load(R / base / f"dino518_{tag}" / "predictions.csv.gz")
            if p is None:
                continue
            orig = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
            J.append(({"claim": f"real-trap crossover ({tag})", "cohort": cohort, "orig": orig}, cross_terms(p)))
    for cohort, c in TRANS.items():
        pa = _load(R / "review2" / "transplant" / f"{c}_dino518" / "predictions.csv.gz")
        pn = _load(R / "review2" / "transplant" / f"{c}_dino518_neutral" / "predictions.csv.gz")
        if pa is not None:
            o = hierarchical_interaction(pa, "mask", "erm", "test_rev", "overlap", 0.0, 1.0, 10000, 20260918, fast=True)
            J.append(({"claim": "transplant interaction (artifact)", "cohort": cohort, "orig": o}, inter_terms(pa)))
            ob = hierarchical_interaction(pa, "balanced", "erm", "test_rev", "overlap", 0.0, 1.0, 10000, 20260918, fast=True)
            J.append(({"claim": "transplant interaction (balanced)", "cohort": cohort, "orig": ob}, inter_terms(pa, arm="balanced")))
            for ov, tag in ((1.0, "in"), (0.0, "out")):
                q = pa[np.isclose(pa.overlap, ov)]
                J.append(({"claim": f"transplant mask - ERM ({tag})", "cohort": cohort, "orig": None},
                          paired_terms(q, "mask", "erm", "test_rev")))
        if pn is not None:
            o = hierarchical_interaction(pn, "mask", "erm", "test_rev", "overlap", 0.0, 1.0, 10000, 20260918, fast=True)
            J.append(({"claim": "transplant interaction (neutral paste)", "cohort": cohort, "orig": o}, inter_terms(pn)))
            for ov, tag in ((1.0, "in"), (0.0, "out")):
                q = pn[np.isclose(pn.overlap, ov)]
                J.append(({"claim": f"neutral mask - ERM ({tag})", "cohort": cohort, "orig": None},
                          paired_terms(q, "mask", "erm", "test_rev")))
            J.append(({"claim": "neutral ERM gap corr - rev (in)", "cohort": cohort, "orig": None},
                      _gap_terms(pn, 1.0)))
            J.append(({"claim": "neutral ERM gap corr - rev (out)", "cohort": cohort, "orig": None},
                      _gap_terms(pn, 0.0)))
        if pa is not None and pn is not None:
            pn2 = pn.copy()
            J.append(({"claim": "artifact - neutral interaction (N3)", "cohort": cohort, "orig": None},
                      inter_terms(pa) + inter_terms(pn2, sign=-1.0)))
    for c, name in NAT.items():
        p = _load(R / "natural" / f"{c}_dino518_repro" / "predictions.csv.gz")
        if p is None:
            continue
        t = p[p.env == "clean"].copy()
        pa1 = t[(t.y == 1) & (t.method == "erm")].artifact_present.mean()
        pa0 = t[(t.y == 0) & (t.method == "erm")].artifact_present.mean()
        hp = 1 if pa1 < pa0 else 0
        hard = ((t.y == 1) & (t.artifact_present == hp)) | ((t.y == 0) & (t.artifact_present == 1 - hp))
        orig = json.loads((R / "natural" / f"{c}_dino518_repro" / "natural_boot.json").read_text())
        for a1, a0 in (("mte", "mask"), ("mte_protect", "mask"), ("balanced", "mask"), ("mte_balanced", "mask")):
            o = orig.get(f"all | {a1}-{a0}")
            J.append(({"claim": f"natural {a1} - {a0} (all pairs)", "cohort": name,
                       "orig": {"seed_delta_mean": o[0], "ci95_lo": o[1], "ci95_hi": o[2]} if o else None},
                      paired_terms(t, a1, a0, "clean")))
        for sub, sel in (("hard", hard), ("all", hard | ~hard)):
            o = orig.get(f"{sub} | mask-erm")
            J.append(({"claim": f"natural mask - ERM ({sub} pairs)", "cohort": name,
                       "orig": {"seed_delta_mean": o[0], "ci95_lo": o[1], "ci95_hi": o[2]} if o else None},
                      paired_terms(t[sel], "mask", "erm", "clean")))
    p = _load(R / "finetune" / "isic" / "resnet50" / "predictions.csv.gz")
    if p is not None:
        orig = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
        J.append(({"claim": "fine-tuned crossover (ResNet-50)", "cohort": "ISIC hair", "orig": orig}, cross_terms(p)))
    return J


def _gap_terms(p, ov):
    """ERM correlated - reversed AUROC at one location: +AUROC(corr) - AUROC(rev) (different test compositions of the
    same images; image weights shared)."""
    q = p[np.isclose(p.overlap, ov) & (p.method == "erm")]
    out = []
    for c, g in q.groupby("seed"):
        for env, coef in (("test_corr", 1.0), ("test_rev", -1.0)):
            z = g[g.env == env]
            out.append((c, coef, z.image_id.astype(str).tolist(), z.y.to_numpy(), z.prob.to_numpy()))
    return out


def _run(j):
    meta, terms = j
    r = crossed_auc_bootstrap(terms, 10000, 20260928 + sum(map(ord, meta["claim"] + meta["cohort"])))
    o = meta.pop("orig")
    row = {**meta, "estimate": r["estimate"], "crossed_lo": r["ci95_lo"], "crossed_hi": r["ci95_hi"],
           "crossed_p": r["p_boot_two_sided"], "crossed_excludes_zero": r["ci_excludes_zero"], "n_images": r["n_images"],
           "n_clusters": r["n_clusters"]}
    if o is not None:
        row.update({"orig_estimate": o.get("seed_delta_mean"), "orig_lo": o.get("ci95_lo"), "orig_hi": o.get("ci95_hi")})
        row["width_ratio"] = (row["crossed_hi"] - row["crossed_lo"]) / (row["orig_hi"] - row["orig_lo"])
    return row


def main():
    J = jobs()
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(_run, J))
    d = pd.DataFrame(rows)
    out = paths.ensure(R / "review3")
    d.to_csv(out / "crossed_ci.csv", index=False)
    print(d.drop(columns=["crossed_p"]).round(3).to_string())


if __name__ == "__main__":
    main()
