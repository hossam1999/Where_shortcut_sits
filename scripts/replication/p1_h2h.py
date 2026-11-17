"""Preliminary per-seed P1 head-to-head (their methods vs our arms a, b, c) on one seed's test images.

Label (A14): "preliminary, 1 seed, no verdict"; P1 head-to-heads are secondary and are not taken to 3 seeds. Paired image-only bootstrap (10,000 replicates; training-seed variance NOT included).
Targets: cardiomegaly Masked (A5: the Standard model scored with test masking); RSNA ActDiff (A11: their RSNA Masked test
input is label-dependent, so it is shown but flagged, never the target).
Their AUROC is reported in their own definition (roc_auc_score on one-hot targets vs both logits, as in their training
log) and as the AUROC of the logit difference, which is the score used for the paired comparisons.

    python scripts/replication/p1_h2h.py --task xray|rsna --seed 1234
Writes results/replication/viviano/preliminary_<task>_<seed>.{csv,md}.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from wtss.replication_stats import paired_bootstrap

D = os.environ.get("WTSS_DATA", "/workspace/data")
ROOT = Path(__file__).resolve().parents[2]
PAPER = {"xray": {"Standard": "0.70 ± 0.05", "Masked": "0.77 ± 0.09", "ActDiff": "0.73 ± 0.03"},
         "rsna": {"Standard": "0.20 ± 0.03", "Masked": "0.56 ± 0.16", "ActDiff": "0.68 ± 0.05"}}


def within_site(df, col):
    v = [roc_auc_score(g.y, g[col]) for _, g in df.groupby("site") if g.y.nunique() == 2]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xray", "rsna"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    t, s = args.task, args.seed
    P = Path(f"{D}/replication/viviano/preds"); A = Path(f"{D}/replication/viviano/arms/{t}_{s}")
    rows, scores = [], {}
    for name, f in (("Standard", f"{t}_standard_{s}"), ("Masked", f"{t}_masked_{s}"), ("ActDiff", f"{t}_actdiff_{s}"),
                    ("ActDiff net on lung-masked images (diagnostic, not a method)", f"{t}_arma_lungs_{s}")):
        p = P / f"{f}.csv.gz"
        if not p.exists():
            continue
        d = pd.read_csv(p); d = d[d.split == "test"].copy()
        d["score"] = d.logit1 - d.logit0
        oh = OneHotEncoder(sparse_output=False).fit_transform(d.y.to_numpy().reshape(-1, 1))
        rows.append({"method": name, "theirs": True, "auroc_their_def": roc_auc_score(oh, d[["logit0", "logit1"]].to_numpy()),
                     "auroc": roc_auc_score(d.y, d.score), "clean_within_site": within_site(d, "score"),
                     "paper": PAPER[t].get(name, "")})
        scores[name] = d.set_index("image_id")[["y", "site", "score"]]
    for arm in ("a_mask_cmc_their_net", "b_dino518_mask_cmc", "c_umte"):
        p = A / f"preds_{arm}.csv.gz"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        rows.append({"method": arm, "theirs": False, "auroc": roc_auc_score(d.y, d.score),
                     "clean_within_site": within_site(d, "score"), "paper": ""})
        scores[arm] = d.set_index("image_id")[["y", "site", "score"]]
    target = "Masked" if t == "xray" else "ActDiff"
    for r in rows:
        for ref in (target, "Standard"):
            if r["method"] == ref or ref not in scores:
                continue
            a, b = scores[r["method"]], scores[ref]
            ids = a.index.intersection(b.index)
            res = paired_bootstrap({s: (list(ids), a.loc[ids, "y"].to_numpy(), a.loc[ids, "score"].to_numpy(),
                                        b.loc[ids, "score"].to_numpy())}, "auroc")
            r[f"delta_vs_{ref}"] = res["delta"]; r[f"ci_vs_{ref}"] = f"[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]"
            r[f"n_paired_vs_{ref}"] = len(ids)
    out = pd.DataFrame(rows)
    rd = ROOT / "results" / "replication" / "viviano"; rd.mkdir(parents=True, exist_ok=True)
    out.to_csv(rd / f"preliminary_{t}_{s}.csv", index=False)
    task = {"xray": "cardiomegaly (site shortcut)", "rsna": "RSNA pneumonia (view shortcut)"}[t]
    md = [f"# P1 {task}, seed {s}: preliminary head-to-head",
          "", "**Preliminary, 1 seed, no verdict** (A14: P1's head-to-heads are secondary — global shortcut — and stay at one "
          "seed). Paired image-only bootstrap (10,000 replicates); training-seed variance is not included.", "",
          f"Target: **{target}**" + (" (A11: their RSNA Masked test input is label-dependent; shown, flagged, not a target)"
                                      if t == "rsna" else " (A5: Standard model scored with test masking)"), "",
          ("Rows: their methods; `a_mask_cmc_their_net` = **registered arm a** (A11/A12: mask_cmc on the ActDiff network's "
           "features of PSPNet-lung-masked images, every split); the 'diagnostic' row is the ActDiff network's own logits on "
           "the same lung-masked images (the feature pass that feeds arm a), not a method; b = DINOv2 + mask_cmc; "
           "c = U-MtE." if t == "rsna" else "Rows: their methods; a = mask_cmc on their Standard/Masked network's circle-masked "
           "features (every split); b = DINOv2 + mask_cmc; c = U-MtE."), "",
          "| method | AUROC (their definition) | AUROC | clean: within-site AUROC | paper | Δ vs " + target + " [95% CI] | Δ vs Standard [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k: f"{r[k]:.3f}" if isinstance(r.get(k), float) and not np.isnan(r[k]) else "—"
        name = r["method"] + (" ⚠ label-dependent test input" if (t == "rsna" and r["method"] == "Masked") else "")
        dt = f"{r[f'delta_vs_{target}']:+.3f} {r[f'ci_vs_{target}']}" if f"delta_vs_{target}" in r else "—"
        ds = f"{r['delta_vs_Standard']:+.3f} {r['ci_vs_Standard']}" if "delta_vs_Standard" in r else "—"
        md.append(f"| {name} | {f('auroc_their_def')} | {f('auroc')} | {f('clean_within_site')} | {r['paper']} | {dt} | {ds} |")
    (rd / f"preliminary_{t}_{s}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
