"""A4 — blinded non-clinician rating (pre-registration A4, Amendments 1-2). Runs only when the passes exist.

  python scripts/round6/analyse_rating.py [--smoke]
Reports, for the 240-image audit package ("main") and the A4b caliper sample ("a4b"):
  intra-rater kappa (pass 1 vs pass 2; presence, and location as a 4-category kappa), rater vs MedGemma,
  and -- after checking each blind key against its committed SHA-256 -- error rates of our cells by cell and
  diagnosis, the differential-error flag, e_A, e_B and the first-order corrected crossover.
The key is read only after both passes of a sheet are saved. Output: results/round6/rating/rating_report.json,
rating_errors.csv, medgemma_informativeness.json (Amendment 2 fallback).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KEYS = {"main": (C.ROOT / "audit_local" / "KEY_open_after_review.csv", C.ROOT / "audit" / "KEY_SHA256.txt"),
        "a4b": (C.ROOT / "results" / "round6" / "_local" / "a4b" / "a4b_key.csv", C.OUT / "a4b_KEY_SHA256.txt")}
PASS = {"main": ("pass1.csv", "pass2.csv"), "a4b": ("a4b_pass1.csv", "a4b_pass2.csv")}
COHORT = {"isic_hair": "isic", "thyroid_calipers": "thyroid", "ovary_calipers": "ovary", "capsule_debris": "capsule",
          "thyroid": "thyroid", "ovary": "ovary"}
KEY_CELL = {"in_roi": "trapA", "out_roi": "trapB", "artifact_free": "artifact_free",
            "trapA": "trapA", "trapB": "trapB"}
LOC = ("inside", "outside", "both", "absent")


def kappa_multi(a, b, cats) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    p0 = float((a == b).mean())
    pe = float(sum((a == k).mean() * (b == k).mean() for k in cats))
    return (p0 - pe) / (1 - pe) if pe < 1 - 1e-12 else float("nan")


def rater_cell(present, location) -> str:
    """Cell implied by the rater: absent -> artifact-free; inside -> Trap A side; outside -> Trap B side."""
    if present == "no" or location == "absent":
        return "artifact_free"
    if present != "yes":
        return "missing"
    return {"inside": "trapA", "outside": "trapB", "both": "mid"}.get(location, "present")


def key_ok(sheet: str):
    key, sha = KEYS[sheet]
    if not key.exists() or not sha.exists():
        return None, "key or its committed SHA-256 not found"
    want = sha.read_text().split()[0]
    got = hashlib.sha256(key.read_bytes()).hexdigest()
    if got != want:
        return None, f"key SHA-256 {got} differs from the committed {want}; not used"
    return pd.read_csv(key), "verified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = C.out_root(a.smoke) / "rating"
    report, err_rows, human_cells = {}, [], []
    for sheet, (f1, f2) in PASS.items():
        p1, p2 = root / f1, root / f2
        if not (p1.exists() and p2.exists()):
            report[sheet] = {"status": "both passes are not saved yet; nothing scored"}
            continue
        r1, r2 = pd.read_csv(p1, dtype=str), pd.read_csv(p2, dtype=str)
        m = r1.merge(r2, on="audit_id", suffixes=("_1", "_2"))
        rep = {"n_both_passes": int(len(m))}
        yn = {"yes": 1, "no": 0}
        x, y = m.artifact_present_1.map(yn), m.artifact_present_2.map(yn)
        ok = x.notna() & y.notna()
        rep["intra_presence_kappa"] = C.kappa_binary(x[ok].astype(int), y[ok].astype(int)) if ok.sum() else None
        l1, l2 = m.artifact_location_1, m.artifact_location_2
        ok = l1.isin(LOC) & l2.isin(LOC)
        rep["intra_location_kappa"] = kappa_multi(l1[ok], l2[ok], LOC) if ok.sum() else None
        # pass 1 is the rating of record; pass 2 measures its repeatability
        m["rater_cell"] = [rater_cell(p, l) for p, l in zip(m.artifact_present_1, m.artifact_location_1)]
        mgf = root / "medgemma.csv"
        if sheet == "main" and mgf.exists():
            g = pd.read_csv(mgf, dtype=str).merge(m[["audit_id", "artifact_present_1", "artifact_location_1"]], on="audit_id")
            xp, gp = g.artifact_present_1.map(yn), g.presence.map(yn)
            ok = xp.notna() & gp.notna()
            rep["rater_vs_medgemma_presence_kappa"] = C.kappa_binary(xp[ok].astype(int), gp[ok].astype(int)) if ok.sum() else None
            xl = g.artifact_location_1.map({"inside": 1, "both": 1, "outside": 0})
            gl = g.location.map(yn)
            ok = xl.notna() & gl.notna()
            rep["rater_vs_medgemma_location_kappa"] = C.kappa_binary(xl[ok].astype(int), gl[ok].astype(int)) if ok.sum() else None
        key, status = key_ok(sheet)
        rep["key"] = status
        if key is not None:
            k = m.merge(key, on="audit_id", how="left")
            k["our_cell"] = k.cell.map(KEY_CELL)
            k["cohort_name"] = k.cohort.map(COHORT) if "cohort" in k else k.cohort_1.map(COHORT)
            k["error"] = [C.contradicts(o, s) if s != "missing" else np.nan for o, s in zip(k.our_cell, k.rater_cell)]
            for (coh, cell, yv), q in k.dropna(subset=["error"]).groupby(["cohort_name", "our_cell", "y"]):
                e = q.error.astype(float).to_numpy()
                ci = C.boot_ci(lambda idx: float(e[idx].mean()), len(e))
                err_rows.append({"sheet": sheet, "cohort": coh, "cell": cell, "y": int(yv), **ci})
            for coh, q in k.dropna(subset=["error"]).groupby("cohort_name"):
                ea = float(((q.our_cell == "trapA") & (q.rater_cell == "trapB")).sum() / max((q.our_cell == "trapA").sum(), 1))
                eb = float(((q.our_cell == "trapB") & (q.rater_cell == "trapA")).sum() / max((q.our_cell == "trapB").sum(), 1))
                flags = {}
                for cell in ("trapA", "trapB"):
                    qq = q[q.our_cell == cell]
                    yy, ee = qq.y.astype(int).to_numpy(), qq.error.astype(float).to_numpy()

                    def stat(idx, yy=yy, ee=ee):
                        e1, e0 = ee[idx][yy[idx] == 1], ee[idx][yy[idx] == 0]
                        return float(e1.mean() - e0.mean()) if len(e1) and len(e0) else np.nan
                    ci = C.boot_ci(stat, len(qq))
                    ci["flagged"] = bool(np.isfinite(ci["estimate"]) and abs(ci["estimate"]) > 0.10 and
                                         (ci["ci95_lo"] > 0 or ci["ci95_hi"] < 0))
                    flags[cell] = ci
                orig = C.original_crossover(coh)
                den = 1 - ea - eb
                flagged = any(f["flagged"] for f in flags.values())
                # A3: withheld for a cohort in which any source flagged differential error (bias_analysis.py)
                bj = C.out_root(a.smoke) / "agreement" / coh / "bias.json"
                other = bool(json.loads(bj.read_text()).get("cohort_flagged")) if bj.exists() else False
                corr = ({"corrected": orig["estimate"] / den, "ci95_lo": orig["ci95_lo"] / den, "ci95_hi": orig["ci95_hi"] / den}
                        if den > 0.05 and not (flagged or other) else
                        {"note": "not reported (differential error flagged in this cohort, by the rating or another "
                                 "source, or 1-e_A-e_B <= 0.05)", "flagged_by_other_source": other})
                rep.setdefault("bias", {})[coh] = {"e_A": ea, "e_B": eb, "differential": flags, "original": orig, **corr}
            human_cells.append(k[["audit_id", "cohort_name", "image_id", "our_cell", "rater_cell"]].assign(sheet=sheet))
        report[sheet] = rep
    # Amendment 2 fallback: a model labeller with no second model labeller is judged against the rater (kappa >= 0.40)
    if human_cells:
        h = pd.concat(human_cells, ignore_index=True)
        inf = {}
        for coh in ("thyroid", "ovary"):
            f = C.out_root(a.smoke) / "matches" / f"{coh}_medgemma.csv"
            if not f.exists():
                continue
            g = pd.read_csv(f, dtype=str)
            q = h[h.cohort_name == coh].merge(g, on="image_id")
            xp = q.rater_cell.map(lambda s: 0 if s == "artifact_free" else (1 if s in ("trapA", "trapB", "mid", "present") else np.nan))
            gp = q.mg_presence.map({"yes": 1, "no": 0})
            ok = xp.notna() & gp.notna()
            kap = C.kappa_binary(xp[ok].astype(int), gp[ok].astype(int)) if ok.sum() else float("nan")
            inf[coh] = {"MedGemma": {"status": "informative" if np.isfinite(kap) and kap >= 0.40 else "uninformative",
                                     "kappa_vs_rater": kap, "n": int(ok.sum())}}
        (root / "medgemma_informativeness.json").write_text(json.dumps(inf, indent=2, default=float))
        h.to_csv(root / "rater_cells.csv", index=False)
    pd.DataFrame(err_rows).to_csv(root / "rating_errors.csv", index=False)
    (root / "rating_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
