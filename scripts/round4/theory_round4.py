"""R11 — prospective test of the linear-Gaussian theory on the R8 / R9 cells (docs/PREREGISTRATION_ROUND4.md).
The fitting code is scripts/analysis/theory_predict.py, unchanged.

  python scripts/round4/theory_round4.py      -> results/round4/theory/{cells.csv, crossovers.csv, slopes.csv, summary.json, SUMMARY.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

VIEWS = ("mask_black", "mask_blur", "crop_box", "crop_mask")


def agree(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    return {"n": int(len(obs)), "sign_agree": int((np.sign(pred) == np.sign(obs)).sum()),
            "pearson_r": float(pearsonr(pred, obs)[0]) if len(obs) > 2 else None, "mae": float(np.abs(pred - obs).mean())}


def main():
    tp = K.load_script("tp", "scripts/analysis/theory_predict.py")
    tp.R = K.OUT.parent
    mv = {(c, "DINOv2"): f"{K.OUT.name}/masking_variants/{c}" for c in K.COHORTS if (K.OUT / "masking_variants" / c / "predictions.csv.gz").exists()}
    dr = {(c, "DINOv2 bins"): f"{K.OUT.name}/dose_response/{c}" for c in K.COHORTS if (K.OUT / "dose_response" / c / "predictions.csv.gz").exists()}
    d = pd.DataFrame(tp.cells("trap", mv) + tp.cells("bins", dr))
    out = K.OUT / "theory"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "cells.csv", index=False)
    T1 = {"theory": tp.summarise(d), "ref_a_rev_eq_clean": tp.summarise(d, "ref_clean"), "ref_b_symmetry": tp.summarise(d, "ref_sym")}
    # T2': crossover per cohort x view, predicted vs observed
    rows = []
    m = d[d.kind == "trap"]
    for (c, arm), q in m.groupby(["cohort", "arm"]):
        if arm not in VIEWS + ("mask",):
            continue
        e = m[(m.cohort == c) & (m.arm == "erm")].set_index("trap")
        v = q.set_index("trap")
        if not {"trapA", "trapB"} <= set(v.index) or not {"trapA", "trapB"} <= set(e.index):
            continue
        cx = lambda col: (v.loc["trapB", col] - e.loc["trapB", col]) - (v.loc["trapA", col] - e.loc["trapA", col])
        rows.append({"cohort": c, "view": arm, "obs": cx("rev"), "pred": cx("pred_rev")})
    cr = pd.DataFrame(rows)
    cr.to_csv(out / "crossovers.csv", index=False)
    prim = cr[cr.view.isin(VIEWS)]
    T2 = agree(prim.pred, prim.obs) if len(prim) else {}
    # T3': bin gains and slopes
    b = d[d.kind == "bins"]
    g = tp.gains(b, "trap") if len(b) else pd.DataFrame()
    srows = []
    for c, q in (g.groupby("cohort") if len(g) else []):
        js = json.loads((K.OUT / "dose_response" / c / "bins_final.json").read_text())
        x = {bb["name"]: bb["x"] for bb in js["bins"]}
        q = q[q.trap.isin(x)].assign(x=lambda t: t.trap.map(x)).sort_values("x")
        if len(q) >= 2:
            xs = q.x.to_numpy()
            w = (xs - xs.mean()) / ((xs - xs.mean()) ** 2).sum()
            srows.append({"cohort": c, "n_bins": len(q), "obs_slope": float(w @ q.obs.to_numpy()), "pred_slope": float(w @ q.pred.to_numpy())})
    sl = pd.DataFrame(srows)
    sl.to_csv(out / "slopes.csv", index=False)
    g.to_csv(out / "bin_gains.csv", index=False)
    T3 = {"bin_gains": agree(g.pred, g.obs) if len(g) else {}, "slopes": agree(sl.pred_slope, sl.obs_slope) if len(sl) else {}}
    t1, ra = T1["theory"], T1["ref_a_rev_eq_clean"]
    quant = bool(T2 and T2["sign_agree"] == T2["n"] and (t1["r"] or 0) >= 0.7 and (T2["pearson_r"] or 0) >= 0.7 and t1["mae"] < ra["mae"])
    verdict = "quantitatively predictive" if quant else "qualitative account"
    js = {"T1": T1, "T2": T2, "T3": T3, "verdict": verdict, "env": K.env_info()}
    (out / "summary.json").write_text(json.dumps(js, indent=1, default=float))
    L = ["# R11 — prospective theory test (fit on clean + correlated, predict reversed; no shared parameter)", "",
         f"- **T1'** reversed AUROC, {t1['n']} new cells: theory MAE {t1['mae']:.3f}, r = {t1['r']:.3f}, "
         f"{100 * t1['within_0.05']:.0f} % within ±0.05; reference (a) rev = clean MAE {ra['mae']:.3f}; "
         f"reference (b) MAE {T1['ref_b_symmetry']['mae']:.3f}."]
    if T2:
        L.append(f"- **T2'** R8 crossovers: signs {T2['sign_agree']}/{T2['n']}, r = {T2['pearson_r']:.3f}, MAE {T2['mae']:.3f}.")
    if T3["slopes"]:
        s = T3["slopes"]
        L.append(f"- **T3'** R9 slopes: signs {s['sign_agree']}/{s['n']}, MAE {s['mae']:.3f}; bin gains: signs "
                 f"{T3['bin_gains']['sign_agree']}/{T3['bin_gains']['n']}, MAE {T3['bin_gains']['mae']:.3f}.")
    L += ["", f"**Verdict by the pre-registered rule: {verdict}.**", ""]
    if len(cr):
        L += ["Crossovers (observed vs predicted):", "", K.md_table(cr.assign(cohort=cr.cohort.map(K.LABEL))), ""]
    if len(sl):
        L += ["Slopes (observed vs predicted):", "", K.md_table(sl.assign(cohort=sl.cohort.map(K.LABEL))), ""]
    (out / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
