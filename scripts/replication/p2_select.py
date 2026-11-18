"""P2 (Pahde 2025) RR-ClArC: choose lambda and layer on P2's validation split (section 3: mean of clean and biased
validation accuracy; the paper leaves both open), then report the chosen model's test accuracy against the target
(section 1: |biased test accuracy - 0.76| <= 0.03; clean 0.86 descriptive). One seed (A3); CIs from an image-only
bootstrap (10,000 replicates), stated in every table. Test rows are read only for the chosen model and for Vanilla.

    python scripts/replication/p2_select.py                 # section 3 runs (their code path: λ {10, 50}, lr 1e-5)
    python scripts/replication/p2_select.py --set a18       # A18 primary: λ 5e9, lr 1e-4, fresh Adam (paper's settings)
Writes results/replication/pahde/rrclarc_selection[_a18].csv and p2_reproduction[_a18].{csv,md}.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
S = Path(f"{D}/replication/pahde/scores")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/replication/pahde"
TARGET, TOL, CLEAN_PAPER = 0.76, 0.03, 0.86


def acc(d):
    return float((d.pred == d.label).mean())


def boot_ci(d, n=10000, seed=20260918):
    rng = np.random.default_rng(seed)
    hit = (d.pred == d.label).to_numpy().astype(float)
    w = rng.poisson(1.0, size=(n, len(hit)))
    v = (w * hit).sum(1) / np.maximum(w.sum(1), 1)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["section3", "a18"], default="section3")
SET = ap.parse_args().set
SFX = "" if SET == "section3" else "_a18"
SC = S if SET == "section3" else Path(f"{D}/replication/pahde/scores_a18")
NRUNS = 6 if SET == "section3" else 3
rows = []
for f in sorted(glob.glob(str(SC / "resnet50d_RRClarc_*.csv.gz"))):
    name = Path(f).name[:-7]
    d = pd.read_csv(f, usecols=["split", "test_set", "label", "pred"])
    v = d[d.split == "val"]
    c, b = acc(v[v.test_set == "clean"]), acc(v[v.test_set == "biased"])
    rows.append({"model": name, "lambda": int(name.split("lamb")[1].split("_")[0]), "layer": name.split("lr0.001_")[1],
                 "val_clean_acc": c, "val_biased_acc": b, "val_mean": (c + b) / 2})
sel = pd.DataFrame(rows).sort_values("val_mean", ascending=False).reset_index(drop=True)
assert len(sel) == NRUNS, len(sel)
sel["chosen"] = sel.index == 0
sel.to_csv(OUT / f"rrclarc_selection{SFX}.csv", index=False)
chosen = sel.model[0]

res = []
for label, name in (("Vanilla (ours, seed 0)", "vanilla_seed0"), (f"RR-ClArC (chosen: {chosen})", chosen)):
    d = pd.read_csv((S if name == "vanilla_seed0" else SC) / f"{name}.csv.gz", usecols=["split", "test_set", "label", "pred"])
    t = d[d.split == "test"]
    for ts in ("clean", "biased"):
        q = t[t.test_set == ts]
        lo, hi = boot_ci(q)
        res.append({"method": label, "test": ts, "n": len(q), "accuracy": acc(q), "ci95_lo": lo, "ci95_hi": hi})
r = pd.DataFrame(res)
r.to_csv(OUT / f"p2_reproduction{SFX}.csv", index=False)
b = float(r[(r.method.str.startswith("RR")) & (r.test == "biased")].accuracy.iloc[0])
verdict = "reproduced" if abs(b - TARGET) <= TOL else "not reproduced"
title = ("section-3 runs, their code path: λ ∈ {10, 50}, effective lr 1e-5 from the loaded optimizer state" if SET == "section3"
         else "A18 PRIMARY: the paper's settings, λ = 5·10⁹ (Tab. 12), lr 1e-4 (p. 31), Vanilla weights only, fresh Adam")
md = [f"# P2 (Pahde et al. 2025) — RR-ClArC reproduction, ISIC 2019, seed 0 — {title}", "",
      "One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not included.", "",
      "## Selection on P2's validation split (section 3: mean of clean and biased accuracy)", "",
      "| λ | layer | val clean | val biased | mean | chosen |", "|---|---|---|---|---|---|"]
md += [f"| {x['lambda']} | {x.layer} | {x.val_clean_acc:.3f} | {x.val_biased_acc:.3f} | {x.val_mean:.3f} | {'**yes**' if x.chosen else ''} |"
       for _, x in sel.iterrows()]
md += ["", "## Test (scored once, chosen model and Vanilla)", "", "| method | test | n | accuracy [95% CI] |", "|---|---|---|---|"]
md += [f"| {x.method} | {x.test} | {x.n} | {x.accuracy:.3f} [{x.ci95_lo:.3f}, {x.ci95_hi:.3f}] |" for _, x in r.iterrows()]
md += ["", f"Target (section 1): RR-ClArC biased-test accuracy 0.76 (paper Table 1), reproduced if |ours − 0.76| ≤ 0.03. "
       f"Ours: {b:.3f} → **{verdict}**. Paper clean accuracy 0.86 (descriptive)."]
(OUT / f"p2_reproduction{SFX}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
