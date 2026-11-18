"""P4 gate (A13 + addendum) and the arm-d split-1 comparison, from per-image test scores (CPU). Machine 2 (A27).

Gate: Segmentation-ERM 3-split mean robust AUROC within 0.03 of the paper (0.727) OR the notebook (0.772), AND Binary-ERM
within 0.03 of the paper (0.359) OR the notebook (0.310) (Binary-ERM: machine 1's scores in git, 0.363). Verdicts only if both
pass. Arm d split 1 (machine 1's scores in git) vs Seg-ERM split 1: paired image-only bootstrap on the same 772 test images.
Writes results/replication/saab/p4_gate.{md,csv} and scores/seg_split<k>.csv.gz. Splits not yet scored are listed as pending.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from wtss.replication_stats import paired_bootstrap
D = os.environ.get("WTSS_DATA", "/workspace/data"); R = ROOT / "results/replication/saab"; (R / "scores").mkdir(parents=True, exist_ok=True)
rows, seg = [], {}
for k in (1, 2, 3):
    e = pd.read_csv(R / f"scores/erm_split{k}.csv.gz")
    rows.append({"method": "Binary-ERM", "split": k, "robust_auroc": roc_auc_score(e.label, e.score), "source": "machine 1 (git)"})
    p = Path(f"{D}/replication/saab/runs/seg_split{k}/preds_test.csv")
    if p.exists():
        d = pd.read_csv(p); seg[k] = d
        pd.DataFrame({"image_id": d.id, "label": d.target, "method": "seg_erm", "test_set": "test", "score": d.score}).to_csv(
            R / f"scores/seg_split{k}.csv.gz", index=False)
        log = pd.read_csv(Path(p).parent / "log.csv")
        rows.append({"method": "Seg-ERM", "split": k, "robust_auroc": roc_auc_score(d.target, d.score), "source": "machine 2",
                     "best_epoch": int(log.valid_dice.idxmax()), "best_val_dice": float(log.valid_dice.max())})
t = pd.DataFrame(rows); t.to_csv(R / "p4_gate.csv", index=False)
md = ["# P4 (Saab et al., MLHC 2022, ISIC) — reproduction gate (A13 + addendum); Seg-ERM on machine 2 (A27)", "",
      "| method | split | robust AUROC | note |", "|---|---|---|---|"]
for _, x in t.iterrows():
    md.append(f"| {x.method} | {x.split} | {x.robust_auroc:.4f} | {x.source}" + (f", best val Dice {x.best_val_dice:.3f} at epoch {x.best_epoch}" if x.method == "Seg-ERM" else "") + " |")
def gate(m, paper, nb):
    v = t[t.method == m].robust_auroc
    if len(v) < 3:
        return f"{m}: {len(v)} of 3 splits scored — pending", None
    mu, sd = v.mean(), v.std(ddof=1); ok = abs(mu - paper) <= 0.03 or abs(mu - nb) <= 0.03
    return (f"{m}: mean {mu:.3f} ± {sd:.3f} (SD over splits) vs paper {paper} (|Δ| {abs(mu - paper):.3f}) / notebook {nb} "
            f"(|Δ| {abs(mu - nb):.3f}) → **{'pass' if ok else 'FAIL'}**"), ok
g1, ok1 = gate("Binary-ERM", 0.359, 0.310); g2, ok2 = gate("Seg-ERM", 0.727, 0.772)
md += ["", "- " + g1, "- " + g2]
if ok1 is not None and ok2 is not None:
    md += ["", f"**P4 gate: {'PASSED — verdicts allowed' if (ok1 and ok2) else 'FAILED — P4 stays descriptive'}** (A13 item 4 + addendum)."]
if 1 in seg:
    a = pd.read_csv(ROOT / "results/replication/armd/scores/p4_split1.csv.gz"); s = seg[1]
    ids = s.id.astype(str).tolist(); ad = a.set_index("image_id").loc[ids]
    b = paired_bootstrap({0: (ids, s.target.to_numpy(), ad.score.to_numpy(), s.score.to_numpy())}, "auroc")
    md += ["", "## Arm d (EfficientNet-B0 + mask_cmc, A20/A21; machine 1's scores) vs Seg-ERM split 1 (paired, same 772 images)",
           f"arm d {b['ours_mean']:.3f} vs Seg-ERM {b['theirs_mean']:.3f}: "
           f"Δ {b['delta']:+.3f} [{b['ci95_lo']:+.3f}, {b['ci95_hi']:+.3f}], p = {b['p_boot_two_sided']:.4f}. One network each, image-only "
           "bootstrap (10,000). Descriptive until the P4 gate."]
(R / "p4_gate.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
