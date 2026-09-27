"""Primary-claim family with Holm (family-wise) and Benjamini–Hochberg correction (docs/STATISTICAL_PLAN.md).

  python scripts/analysis/primary_claims.py      # -> results/PRIMARY_CLAIMS.csv / .md
All tests: reversed-test AUROC, Trap A unless stated, hierarchical paired bootstrap (10,000), two-sided p.
"""
from __future__ import annotations

import pandas as pd

from wtss import paths
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, slim

R = paths.RESULTS
# (family, claim label, results dir, kind, a1, a0, trap)
CLAIMS = [
    ("P1 location crossover", "ISIC hair (DINOv2)", "spec_e13/dino518_spec", "cross", "mask", "erm", None),
    ("P1 location crossover", "Thyroid (DINOv2)", "thyroid/dino518_main", "cross", "mask", "erm", None),
    ("P1 location crossover", "Capsule (DINOv2)", "capsule/dino518_main", "cross", "mask", "erm", None),
    ("P1 location crossover", "Ovary (DINOv2)", "ovary/dino518_main", "cross", "mask", "erm", None),
    ("P2 U-MtE (protected) > mask", "ISIC hair", "spec_e13/dino518_spec_protect_generic", "pair", "mte_protect", "mask", "trapA"),
    ("P2 U-MtE (protected) > mask", "Thyroid", "thyroid/dino518_protect_generic", "pair", "mte_protect", "mask", "trapA"),
    ("P2 U-MtE (protected) > mask", "Capsule", "capsule/dino518_protect_generic", "pair", "mte_protect", "mask", "trapA"),
    ("P2 U-MtE (protected) > mask", "Ovary", "ovary/dino518_universal", "pair", "mte_protect", "mask", "trapA"),
    ("P3 erase > augment", "ISIC hair", "spec_e13/dino518_spec_ablation", "pair", "mte", "mte_aug", "trapA"),
    ("P3 erase > augment", "Thyroid", "thyroid/dino518_ablation", "pair", "mte", "mte_aug", "trapA"),
    ("P3 erase > augment", "Capsule", "capsule/dino518_universal", "pair", "mte", "mte_aug", "trapA"),
    ("P3 erase > augment", "Ovary", "ovary/dino518_universal", "pair", "mte", "mte_aug", "trapA"),
    ("P4 U-MtE+balanced > balanced", "ISIC hair", "spec_e13/dino518_spec_universal", "pair", "mte_balanced", "balanced", "trapA"),
    ("P4 U-MtE+balanced > balanced", "Thyroid", "thyroid/dino518_universal", "pair", "mte_balanced", "balanced", "trapA"),
    ("P4 U-MtE+balanced > balanced", "Capsule", "capsule/dino518_universal", "pair", "mte_balanced", "balanced", "trapA"),
    ("P4 U-MtE+balanced > balanced", "Ovary", "ovary/dino518_universal", "pair", "mte_balanced", "balanced", "trapA"),
]


def resolve(rel):
    """Historical runs were split over several tags; arms are deterministic (identical predictions across tags,
    verified), so a fresh reproduction (Makefile) stores them in one consolidated *_universal / spec_universal run."""
    if (R / rel / "predictions.csv.gz").exists():
        return rel
    cohort, tag = rel.split("/", 1)
    alt = f"{cohort}/dino518_spec_universal" if cohort == "spec_e13" else f"{cohort}/dino518_universal"
    return alt if (R / alt / "predictions.csv.gz").exists() else rel


def holm(p):
    import numpy as np
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def bh(p):
    import numpy as np
    p = np.asarray(p); o = np.argsort(p)[::-1]; m = len(p); adj = np.empty(m); run = 1.0
    for r, i in enumerate(o):
        run = min(run, p[i] * m / (m - r)); adj[i] = run
    return adj


def main():
    rows = []
    for fam, lab, rel, kind, a1, a0, trap in CLAIMS:
        rel = resolve(rel)
        f = R / rel / "predictions.csv.gz"
        if not f.exists():
            print("missing", rel); continue
        p = pd.read_csv(f)
        if kind == "cross":
            r = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], a1, a0, "test_rev", 10000, 11)
        else:
            q = p[p.trap == trap]
            r = hierarchical_paired_bootstrap(slim(q, "test_rev", (a1, a0)), a1, a0, "test_rev", 10000, 29, fast=True)
        rows.append({"family": fam, "cohort": lab, "estimate": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"],
                     "ci95_hi": r["ci95_hi"], "p": r["p_boot_two_sided"], "source": rel})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d["p_holm"] = holm(d.p.to_numpy())
    d["p_bh"] = bh(d.p.to_numpy())
    d["supported_holm_0.05"] = (d.p_holm < 0.05) & (d.estimate > 0)
    d.to_csv(R / "PRIMARY_CLAIMS.csv", index=False)
    md = ["# Primary claims (Holm family-wise over all rows; BH for reference)\n",
          f"Supported after Holm (correct direction): {int(d['supported_holm_0.05'].sum())}/{len(d)}\n",
          "| family | cohort | estimate [95 % CI] | p | p (Holm) | p (BH) |", "|---|---|---|---|---|---|"]
    for r in d.itertuples():
        md.append(f"| {r.family} | {r.cohort} | {r.estimate:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] | {r.p:.4f} | "
                  f"{r.p_holm:.4f}{(' ✓' if r.estimate > 0 else ' ✗ (opposite direction)') if r.p_holm < 0.05 else ''} | {r.p_bh:.4f} |")
    (R / "PRIMARY_CLAIMS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
