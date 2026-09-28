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
    "Mendeley_vmxhn95j8z": "cell_expert", "BUSClean": "cell_bus",
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
        c_obs = observed.get("estimate")
        if c_obs is None and observed.get("original"):
            c_obs = observed["original"].get("estimate")
            c_lo, c_hi = observed["original"]["ci95_lo"], observed["original"]["ci95_hi"]
            c_from = "original"
        else:
            c_lo, c_hi = observed.get("ci95_lo"), observed.get("ci95_hi")
            c_from = "cleaned" if observed.get("ran") else "missing"
        flagged_any = False
        e_a = e_b = float("nan")
        for source, col in SOURCE_CELLS.items():
            if col not in per.columns or gates.get((name, source)) != "analysed":
                continue
            if col == "cell_bus":
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
            diff = _diff_flag(per.y.to_numpy()[sub], err[sub])
            diff.update({"cohort": name, "source": source})
            flagged_any = flagged_any or diff["flagged"]
            rows.append({"cohort": name, "source": source, "cell": "diff_y1_minus_y0", "y": "diff", "n": int(sub.sum()),
                         "error_rate": diff["estimate"], "ci95_lo": diff["ci95_lo"], "ci95_hi": diff["ci95_hi"],
                         "flagged": diff["flagged"]})
            ea, na = _opposite_rate(per.cell, src, "trapA", "trapB")
            eb, nb = _opposite_rate(per.cell, src, "trapB", "trapA")
            if np.isfinite(ea):
                e_a = ea
            if np.isfinite(eb):
                e_b = eb
            rows.append({"cohort": name, "source": source, "cell": "e_A", "y": "all", "n": na, "error_rate": ea})
            rows.append({"cohort": name, "source": source, "cell": "e_B", "y": "all", "n": nb, "error_rate": eb})
        corr = {}
        if c_obs is not None and np.isfinite(e_a) and np.isfinite(e_b) and not flagged_any:
            den = 1 - e_a - e_b
            if den > 0.05:
                corr = {"c_obs": c_obs, "c_from": c_from, "e_A": e_a, "e_B": e_b,
                        "corrected": c_obs / den,
                        "ci95_lo": None if c_lo is None else c_lo / den,
                        "ci95_hi": None if c_hi is None else c_hi / den,
                        "note": "First-order attenuation. Reported because differential error was not flagged."}
            else:
                corr = {"note": "1 - e_A - e_B <= 0.05; correction not reported"}
        elif flagged_any:
            corr = {"note": "Differential error flagged; the cohort conclusion rests on A2. Correction not reported.",
                    "e_A": e_a, "e_B": e_b}
        else:
            corr = {"note": "No analysed source with an opposite-side rate, or no crossover.", "e_A": e_a, "e_B": e_b}
        (root / "agreement" / name).mkdir(parents=True, exist_ok=True)
        (root / "agreement" / name / "bias.json").write_text(json.dumps(corr, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(root / "bias.csv", index=False)
    print(df.head(20).to_string(index=False) if len(df) else "no analysed sources")


if __name__ == "__main__":
    main()
