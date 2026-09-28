"""Analyse the filled image-audit sheet(s) and write the results into Stage 6.

  python audit/analyse_audit.py audit/review_sheet_filled.csv [second_reviewer.csv]

Per cohort: precision and recall of our artifact labels against the reviewer (artifact_present yes/no; our label =
in-ROI or out-of-ROI cell -> present, artifact-free cell -> absent), agreement of the location (reviewer
inside/outside/both/none vs our cell), share of correct automatic artifact masks and of correct ROI masks, and Cohen's
kappa between two reviewers when a second sheet is given. 95 % CIs by bootstrap over images (2,000 replicates).
Outputs: results/audit/audit_results.csv and stages/stage6_remedies_audit_paper/tables/audit_results.tex (which replaces
the placeholder in Stage 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELL_LOC = {"in_roi": "inside", "out_roi": "outside", "artifact_free": "none"}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def norm(s):
    return s.astype(str).str.strip().str.lower().replace({"nan": ""})


def analyse(sheet: Path, second: Path | None = None) -> pd.DataFrame:
    key = pd.read_csv(ROOT / "audit_local" / "KEY_open_after_review.csv")  # regenerate with make_audit_sample.py if missing
    d = key.merge(pd.read_csv(sheet), on=["audit_id", "cohort"])
    for c in ("artifact_present", "artifact_location", "mask_correct", "roi_mask_correct"):
        d[c] = norm(d[c])
    d = d[d.artifact_present.isin(["yes", "no"])]
    if second is not None:
        s2 = pd.read_csv(second)[["audit_id", "artifact_present", "artifact_location"]]
        s2.columns = ["audit_id", "present_2", "location_2"]
        d = d.merge(s2, on="audit_id", how="left")
        d["present_2"], d["location_2"] = norm(d.present_2), norm(d.location_2)
    rows = []
    for cohort, q in d.groupby("cohort"):
        q = q.reset_index(drop=True)
        truth = (q.artifact_present == "yes").to_numpy().astype(int)
        auto = (q.cell != "artifact_free").to_numpy().astype(int)
        prec = lambda i: truth[i][auto[i] == 1].mean() if (auto[i] == 1).any() else np.nan
        rec = lambda i: auto[i][truth[i] == 1].mean() if (truth[i] == 1).any() else np.nan
        loc_ok = (q.artifact_location == q.cell.map(CELL_LOC)).to_numpy().astype(float)
        present_both = (truth == 1) & (auto == 1)
        locf = lambda i: loc_ok[i][present_both[i]].mean() if present_both[i].any() else np.nan
        mask_ok = (q.mask_correct == "yes").to_numpy().astype(float)
        roi_ok = (q.roi_mask_correct == "yes").to_numpy().astype(float)
        idx = np.arange(len(q))
        r = {"cohort": cohort, "n_rated": len(q), "precision": prec(idx), "recall": rec(idx),
             "location_agreement": locf(idx), "artifact_mask_correct": mask_ok[auto == 1].mean() if (auto == 1).any() else np.nan,
             "roi_mask_correct": roi_ok.mean(), "roi_mask_partly": float((q.roi_mask_correct == "partly").mean())}
        for k, f in (("precision", prec), ("recall", rec), ("location_agreement", locf)):
            r[f"{k}_lo"], r[f"{k}_hi"] = boot(f, len(q))
        if second is not None and q.present_2.isin(["yes", "no"]).any():
            m = q.present_2.isin(["yes", "no"])
            r["kappa_presence"] = kappa(q.artifact_present[m], q.present_2[m])
            both = m & (q.artifact_present == "yes") & (q.present_2 == "yes")
            r["kappa_location"] = kappa(q.artifact_location[both], q.location_2[both]) if both.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def to_tex(res: pd.DataFrame) -> str:
    f = lambda v, lo, hi: f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    L = ["\\begin{table}[h]\\centering\\small",
         "\\caption{Image audit of the automatic labels (reviewer vs.\\ our labels; 95\\% CIs by bootstrap over images).}"
         "\\label{tab:audit}", "\\begin{tabular}{lrcccc}\\toprule",
         "Cohort & $n$ & Precision & Recall & Location agreement & ROI mask correct \\\\\\midrule"]
    for r in res.itertuples():
        L.append(f"{r.cohort.replace('_', ' ')} & {r.n_rated} & {f(r.precision, r.precision_lo, r.precision_hi)} & "
                 f"{f(r.recall, r.recall_lo, r.recall_hi)} & {f(r.location_agreement, r.location_agreement_lo, r.location_agreement_hi)} & "
                 f"{r.roi_mask_correct:.3f} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    if "kappa_presence" in res:
        L.append("Inter-reviewer agreement (Cohen's $\\kappa$, presence): " + ", ".join(
            f"{r.cohort.replace('_', ' ')} {r.kappa_presence:.3f}" for r in res.itertuples()) + ".")
    return "\n".join(L) + "\n"


def main():
    sheet = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "audit" / "review_sheet.csv"
    second = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    res = analyse(sheet, second)
    if res.empty:
        print("no rated rows yet (fill artifact_present with yes/no)")
        return
    out = ROOT / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "audit_results.csv", index=False)
    t = ROOT / "stages" / "stage6_remedies_audit_paper" / "tables"
    t.mkdir(parents=True, exist_ok=True)
    (t / "audit_results.tex").write_text(to_tex(res))
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
