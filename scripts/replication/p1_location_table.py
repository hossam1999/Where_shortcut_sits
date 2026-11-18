"""P1 location test table (amendments A15, A16): Standard, Masked (theirs) and our arms a, b, c at both tube positions.
Per position and test set: AUROC; on the reversed test: gain over Standard, paired difference vs Masked (95% CI) and the
crossover = gain outside - gain inside (95% CI). Image-only bootstrap (one seed, A3), 10,000 replicates; the test images
are the same at both positions, so one Poisson weight per image is shared across positions and methods.

    python scripts/replication/p1_location_table.py              # registered (A15/A16), primary
    python scripts/replication/p1_location_table.py --model last # A17, POST HOC: their epoch-100 models (arms b, c reused)
Writes results/replication/viviano/location_1234.{csv,md} (or location_1234_posthoc_last.{csv,md}).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wtss.replication_stats import _metric, paired_bootstrap
from wtss.stats import _ci, boot_p

D = os.environ.get("WTSS_DATA", "/workspace/data")
L = Path(f"{D}/replication/viviano/location")
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("a_mask_cmc_their_net", "b_dino518_mask_cmc", "c_umte")
ARTIFACT = ("Artifact: a synthetic bar-shaped radio-opaque marker (43 x 6 px at 224 px; straight, bright, with a central "
            "stripe), not a realistic tube. Outside placements lie near the image edge: below the lungs + heart near the "
            "bottom edge (47.5% of test images), above the lungs near the top edge (38.2%) or beside the ROI (14.3%). "
            "Realism is covered elsewhere in the thesis (real device traps on NIH / RANZCR; transplant of real artifacts).")
NAMES = {"standard": "Standard (theirs)", "masked": "Masked (theirs, A5)", "a_mask_cmc_their_net": "a: mask_cmc on their net",
         "b_dino518_mask_cmc": "b: DINOv2 + mask_cmc", "c_umte": "c: U-MtE"}


def load(pos, sfx=""):
    s = pd.read_csv(L / pos / f"scores{sfx}.csv.gz")
    s = s[s.test_set.isin(["corr", "rev", "clean"])][["image_id", "label", "method", "test_set", "score"]]
    parts = [s]
    for arm in ARMS:   # A17: only arm a uses their network; b and c are reused from the registered run
        f = L / pos / (f"arms{sfx}" if arm.startswith("a_") else "arms") / f"preds_{arm}.csv.gz"
        if f.exists():
            parts.append(pd.read_csv(f)[["image_id", "label", "method", "test_set", "score"]])
    return pd.concat(parts)


def crossover(sc, method, n_boot=10000, seed=20260918):
    """(AUC_m,out - AUC_std,out) - (AUC_m,in - AUC_std,in) on the reversed test, shared image weights."""
    get = lambda pos, m: sc[pos][(sc[pos].method == m) & (sc[pos].test_set == "rev")].set_index("image_id").sort_index()
    ids = get("inside", "standard").index
    f = {(p, m): _metric("auroc", get(p, m).loc[ids, "label"].to_numpy(), get(p, m).loc[ids, "score"].to_numpy())
         for p in ("inside", "outside") for m in (method, "standard")}
    point = (f["outside", method][1] - f["outside", "standard"][1]) - (f["inside", method][1] - f["inside", "standard"][1])
    rng = np.random.default_rng(seed); out = []
    for b0 in range(0, n_boot, 250):
        Wt = rng.poisson(1.0, size=(min(250, n_boot - b0), len(ids))).astype(float)
        g = {k: v[0](Wt) for k, v in f.items()}
        out.append((g["outside", method] - g["outside", "standard"]) - (g["inside", method] - g["inside", "standard"]))
    arr = np.concatenate(out)
    lo, hi = _ci(arr)
    return point, lo, hi, boot_p(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["best", "last"], default="best")
    sfx = "" if ap.parse_args().model == "best" else "_last"
    ph = "POST HOC " if sfx else ""
    sc = {p: load(p, sfx) for p in ("inside", "outside") if (L / p / f"scores{sfx}.csv.gz").exists()}
    circ = pd.read_csv(ROOT / "results/replication/viviano/location_tube_vs_circle.csv").set_index("position")
    rows = []
    for pos, s in sc.items():
        for m in ["standard", "masked", *ARMS]:
            if m not in set(s.method):
                continue
            r = {"position": pos, "method": m,
                 "tube_pixels_kept_by_their_circle": float(circ.loc[pos, "mean share of tube pixels inside their circle"]),
                 "tube_pixels_kept_by_lungs_heart": float(circ.loc[pos, "mean share of tube pixels inside lungs+heart"])}
            for t in ("corr", "rev", "clean"):
                q = s[(s.method == m) & (s.test_set == t)]
                r[f"{t}_auroc"] = float(_metric("auroc", q.label.to_numpy(), q.score.to_numpy())[1])
            ref = {k: s[(s.method == k) & (s.test_set == "rev")].set_index("image_id").sort_index() for k in ("standard", "masked")}
            me = s[(s.method == m) & (s.test_set == "rev")].set_index("image_id").sort_index()
            r["rev_gain_over_standard"] = r["rev_auroc"] - float(_metric("auroc", ref["standard"].label.to_numpy(), ref["standard"].score.to_numpy())[1])
            if m != "standard":   # gain over Standard with its paired CI (needed for E-L1 to E-L3)
                ids = me.index
                res = paired_bootstrap({0: (list(ids), me.label.to_numpy(), me.score.to_numpy(), ref["standard"].loc[ids, "score"].to_numpy())}, "auroc")
                r["rev_gain_over_standard_ci"] = f"[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]"
            if m != "masked":
                ids = me.index
                res = paired_bootstrap({0: (list(ids), me.label.to_numpy(), me.score.to_numpy(), ref["masked"].loc[ids, "score"].to_numpy())}, "auroc")
                r.update(rev_delta_vs_masked=res["delta"], rev_delta_vs_masked_ci=f"[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]")
            rows.append(r)
    t = pd.DataFrame(rows)
    if len(sc) == 2:
        for m in ["masked", *ARMS]:
            if m in set(sc["inside"].method) and m in set(sc["outside"].method):
                pt, lo, hi, p = crossover(sc, m)
                t.loc[t.method == m, "crossover"] = pt
                t.loc[t.method == m, "crossover_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
    if sfx:
        t.insert(0, "analysis", "POST HOC (A17): their epoch-100 models; arms b, c reused")
    stem = "location_1234_posthoc_last" if sfx else "location_1234"
    rd = ROOT / "results/replication/viviano"; t.to_csv(rd / f"{stem}.csv", index=False)
    md = ["# P1 location test (A15/A16): marker inside vs outside the lungs + heart ROI, P1's pipeline, seed 1234", ""] if not sfx else [
          "# POST HOC sensitivity analysis (A17) — P1 location test on their epoch-100 models (`last_model_1234.pth.tar`)", "",
          "**Every number in this table is POST HOC** (decided after the registered table was seen). The registered table "
          "(`location_1234.md`, their selection rule: epoch 11 inside, epoch 8 outside) is the primary result; this one is "
          "descriptive only, with no verdict. Standard, Masked and arm a use their epoch-100 network at both positions; arms b "
          "and c do not use their network and are the registered predictions, reused unchanged.", ""]
    md += [ARTIFACT, "",
          "One seed: paired image-only bootstrap (10,000 replicates); training-seed variance is not included (A3). "
          "Crossover = gain over Standard outside − gain inside (reversed test). 'Kept by circle' = mean share of marker pixels "
          "inside P1's circle mask (what Masked and arm a keep); 'kept by lungs + heart' = the same for the PSPNet lungs ∪ heart mask "
          "(what arms b and c keep).", "",
          "| position | method | kept by circle | kept by lungs + heart | correlated | reversed | clean | reversed gain vs Standard [95% CI] | Δ vs Masked (reversed) [95% CI] | crossover [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}" if isinstance(v, float) else str(v)
    for _, r in t.iterrows():
        dv = f"{r.get('rev_delta_vs_masked', np.nan):+.3f} {r.get('rev_delta_vs_masked_ci', '')}" if isinstance(r.get("rev_delta_vs_masked_ci"), str) else "—"
        cx = f"{r['crossover']:+.3f} {r['crossover_ci']}" if "crossover_ci" in r and isinstance(r.get("crossover_ci"), str) else "—"
        md.append(f"| {ph}{r.position} | {NAMES[r.method]} | {r.tube_pixels_kept_by_their_circle:.3f} | {r.tube_pixels_kept_by_lungs_heart:.3f} | {r.corr_auroc:.3f} | {r.rev_auroc:.3f} | "
                  f"{r.clean_auroc:.3f} | {f(r.rev_gain_over_standard)} {r.get('rev_gain_over_standard_ci', '') if isinstance(r.get('rev_gain_over_standard_ci'), str) else ''} | {dv} | {cx} |")
    (rd / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
