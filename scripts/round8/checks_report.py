"""Round 8, registered checks of docs/PREREGISTRATION_ROUND8.md section 6 (reporting only; no analysis is changed).

  python scripts/round8/checks_report.py [--smoke]   -> results/round8/checks_summary.md (appended to SUMMARY.md)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("round8_common", Path(__file__).resolve().parent / "common.py")
C = importlib.util.module_from_spec(spec); sys.modules.setdefault("round8_common", C); spec.loader.exec_module(C)
R4 = C.load_round4_common()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = C.out_dir(a.smoke)
    lines = ["", "## Registered checks (section 6)", "", "### Pasted-versus-real probe (DINOv2)", ""]
    rows = []
    for f in sorted((out / "checks").glob("locrand_probe_*.json")):
        d = json.loads(f.read_text())
        rows.append({"cohort": d["cohort"], "pasted_vs_real AUROC": f"{d.get('pasted_vs_real_auroc', float('nan')):.3f} "
                     f"[{d.get('pasted_vs_real_ci95_lo', float('nan')):.3f}, {d.get('pasted_vs_real_ci95_hi', float('nan')):.3f}]",
                     "reference plain_vs_real AUROC": f"{d.get('reference_plain_vs_real_auroc', float('nan')):.3f}",
                     "distinguishable (> 0.75)": d.get("distinguishable")})
    lines.append(R4.md_table(pd.DataFrame(rows)))
    if any(r["distinguishable (> 0.75)"] for r in rows):
        bad = ", ".join(r["cohort"] for r in rows if r["distinguishable (> 0.75)"])
        lines += ["", f"**Pasted artifacts are distinguishable from real ones ({bad}): the locrand / locrand_loc results "
                      "may not transfer to real artifacts.**"]
    lines += ["", "### Placements", ""]
    pr = [json.loads(f.read_text()) for f in sorted((out / "checks").glob("locrand_placements_*.json"))]
    lines.append(R4.md_table(pd.DataFrame(pr)))
    # paste balance and fallback rates from the confirmation choices
    bal, fb = [], []
    for f in sorted((out / "confirm").glob("*/*/choices.csv")):
        ch = pd.read_csv(f)
        if "family" not in ch:  # fine-tuned runs: selection rows without families
            continue
        coh = f"{f.parent.parent.name}/{f.parent.name}"
        pl = ch[ch.setting == "_plan"]
        for fam, g in pl.groupby("family"):
            gap = (g.p_a_y1_after - g.p_a_y0_after).abs()
            bal.append({"run": coh, "family": fam, "training sets": len(g), "mean |P(A|Y=1)-P(A|Y=0)| after": gap.mean(),
                        "share > 0.02": (gap > 0.02).mean(), "mean pastes": (g.n_pasted_y0 + g.n_pasted_y1).mean()})
        sel = ch[(ch.get("chosen") == True) & (ch.setting != "_plan")]  # noqa: E712
        for fam, g in sel.groupby("family"):
            fb.append({"run": coh, "family": fam, "training sets": len(g), "fallback to masking": (g.setting == "mask").mean()})
    if bal:
        lines += ["", "### Paste balance (confirmation training sets)", "", R4.md_table(pd.DataFrame(bal))]
    if fb:
        f = pd.DataFrame(fb)
        f = f[f.family.isin(["mask_bal", "mask_cmc", "full_cmc", "locrand_loc", "locrand"])]
        lines += ["", "### Fallback rates of the selection rule (confirmation)", "", R4.md_table(f)]
    text = "\n".join(lines) + "\n"
    (out / "checks_summary.md").write_text(text)
    s = out / "SUMMARY.md"
    if s.exists() and "## Registered checks" not in s.read_text():
        s.write_text(s.read_text() + text)
    C.log(written=str(out / "checks_summary.md"))


if __name__ == "__main__":
    main()
