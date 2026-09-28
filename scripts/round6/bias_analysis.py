"""A3 — differential error and the attenuation correction.

  python scripts/round6/bias_analysis.py
  python scripts/round6/bias_analysis.py --smoke
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SOURCE_CELLS = {
    "DermArtifactDB": "cell_derm", "IMA++": "cell_ima", "Kabir": "cell_kabir",
    "Mendeley_vmxhn95j8z": "cell_expert", "BUSClean": "cell_bus", "MedGemma": "cell_mg",
}


def _diff_flag(y, err):
    y, err = np.asarray(y), np.asarray(err, float)

    def stat(idx):
        e1 = err[idx][y[idx] == 1]
        e0 = err[idx][y[idx] == 0]
        if len(e1) == 0 or len(e0) == 0:
            return np.nan
        return float(e1.mean() - e0.mean())

    ci = C.boot_ci(stat, len(y))
    flagged = bool(np.isfinite(ci["estimate"]) and abs(ci["estimate"]) > 0.10 and (ci["ci95_lo"] > 0 or ci["ci95_hi"] < 0))
    ci["flagged"] = flagged
    return ci


def _opposite_rate(cell, src, want_our, want_src):
    m = (cell == want_our) & (src == want_src)
    base = cell == want_our
    known = src.isin(["trapA", "trapB", "mid", "artifact_free", "present"])
    den = int((base & known).sum())
    return (int(m.sum()) / den) if den else float("nan"), den


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = C.out_root(a.smoke)
    cov = pd.read_csv(root / "coverage.csv")
    gates = {(r.cohort, r.source): r.gate for r in cov[cov.cell == "all"].itertuples(index=False)}
    rows = []
    for name in C.COHORTS:
        per_p = root / "agreement" / name / "per_image.csv"
        if not per_p.exists():
            continue
        per = pd.read_csv(per_p)
        cross_p = root / "clean_traps" / name / "crossover.json"
        observed = json.loads(cross_p.read_text()) if cross_p.exists() else {}
        # the error rates are measured on the original labels, so the correction applies to the original (Stage 3)
        # crossover; the cleaned crossover (A2) is its empirical counterpart and is not corrected again
        orig = observed.get("original") or C.original_crossover(name)
        c_obs, c_lo, c_hi, c_from = orig.get("estimate"), orig.get("ci95_lo"), orig.get("ci95_hi"), "original (Stage 3)"
        informative = json.loads((root / "agreement" / name / "agreement.json").read_text()).get("informative", {}) \
            if (root / "agreement" / name / "agreement.json").exists() else {}
        per_source = {}
        for source, col in SOURCE_CELLS.items():
            if col not in per.columns or gates.get((name, source)) != "analysed":
                continue
            if source in ("BUSClean", "MedGemma") and informative.get(source, {}).get("status") != "informative":
                continue
            src = per[col].astype(str)
            known = ~src.isin(["missing", "nan"])
            err = np.array([C.contradicts(o, s) for o, s in zip(per.cell, src)])
            sub = known.to_numpy()
            if sub.sum() == 0:
                continue
            for cell in ("trapA", "trapB", "artifact_free", "mid"):
                for yv in (0, 1):
                    m = sub & (per.cell == cell).to_numpy() & (per.y == yv).to_numpy()
                    rows.append({"cohort": name, "source": source, "cell": cell, "y": yv,
                                 "n": int(m.sum()), "error_rate": float(err[m].mean()) if m.sum() else float("nan")})
            # differential error by trap and diagnosis (registered): Y=1 minus Y=0 within each trap cell
            flagged = False
            for cell in ("trapA", "trapB"):
                m = sub & (per.cell == cell).to_numpy()
                if m.sum() == 0:
                    continue
                diff = _diff_flag(per.y.to_numpy()[m], err[m])
                flagged = flagged or diff["flagged"]
                rows.append({"cohort": name, "source": source, "cell": f"diff_y1_minus_y0_{cell}", "y": "diff",
                             "n": int(m.sum()), "error_rate": diff["estimate"], "ci95_lo": diff["ci95_lo"],
                             "ci95_hi": diff["ci95_hi"], "flagged": diff["flagged"]})
            ea, na = _opposite_rate(per.cell, src, "trapA", "trapB")
            eb, nb = _opposite_rate(per.cell, src, "trapB", "trapA")
            rows.append({"cohort": name, "source": source, "cell": "e_A", "y": "all", "n": na, "error_rate": ea})
            rows.append({"cohort": name, "source": source, "cell": "e_B", "y": "all", "n": nb, "error_rate": eb})
            if not (np.isfinite(ea) and np.isfinite(eb)):
                per_source[source] = {"note": "source has no location information for both traps", "e_A": ea, "e_B": eb}
            elif flagged:
                per_source[source] = {"note": "Differential error flagged; the cohort conclusion rests on A2. Correction not "
                                              "reported.", "e_A": ea, "e_B": eb, "flagged": True}
            elif c_obs is None:
                per_source[source] = {"note": "no crossover", "e_A": ea, "e_B": eb}
            else:
                den = 1 - ea - eb
                per_source[source] = ({"c_obs": c_obs, "c_from": c_from, "e_A": ea, "e_B": eb, "n_A": na, "n_B": nb,
                                       "corrected": c_obs / den, "ci95_lo": None if c_lo is None else c_lo / den,
                                       "ci95_hi": None if c_hi is None else c_hi / den,
                                       "note": "First-order attenuation (approximation); reported because differential "
                                               "error was not flagged."}
                                      if den > 0.05 else {"note": "1 - e_A - e_B <= 0.05; correction not reported",
                                                          "e_A": ea, "e_B": eb})
        corr = {"by_source": per_source}
        (root / "agreement" / name).mkdir(parents=True, exist_ok=True)
        (root / "agreement" / name / "bias.json").write_text(json.dumps(corr, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(root / "bias.csv", index=False)
    print(df.head(20).to_string(index=False) if len(df) else "no analysed sources")


if __name__ == "__main__":
    main()
