"""Write results/round6/SUMMARY.md from the tables the other scripts produced."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _gpu() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    except Exception:
        return "unknown"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = C.out_root(a.smoke)
    lines = ["# Round 6", "", f"Commit `{_commit()}`. GPU: {_gpu()}.", ""]
    lines.append("## Sources and coverage")
    src = root / "sources.json"
    if src.exists():
        d = json.loads(src.read_text())
        lines.append("| source | licence | matched | note |")
        lines.append("|---|---|---|---|")
        for s in d.get("sources", []):
            lines.append(f"| {s.get('name')} | {s.get('licence')} | {s.get('n_matched', '')} | {s.get('skip_reason') or s.get('note') or ''} |")
        lines.append("")
        if (root / "coverage.csv").exists():
            cov = pd.read_csv(root / "coverage.csv")
            top = cov[cov.cell == "all"][["cohort", "source", "n", "gate"]]
            lines.append(C.R4.md_table(top))
            lines.append("")
            lines.append("Gate (Amendment 1): ≥200 analysed as registered, 50–199 descriptive only, <50 not feasible.")
    lines.append("")
    lines.append("## Agreement")
    for name in C.COHORTS:
        p = root / "agreement" / name / "agreement.json"
        if not p.exists():
            continue
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(p.read_text().strip())
        lines.append("```")
        lines.append("")
    lines.append("## Cleaned traps (HA2)")
    sp = root / "clean_traps" / "SUMMARY.csv"
    if sp.exists():
        lines.append(C.R4.md_table(pd.read_csv(sp)))
        lines.append("")
    lines.append("## Differential error and attenuation (A3)")
    bp = root / "bias.csv"
    if bp.exists() and bp.stat().st_size > 0:
        b = pd.read_csv(bp)
        if len(b):
            flag = b[b.cell.astype(str).str.startswith(("diff_y1_minus_y0", "e_A", "e_B"))] if "cell" in b else b
            lines.append(C.R4.md_table(flag))
            lines.append("")
    for name in C.COHORTS:
        bj = root / "agreement" / name / "bias.json"
        aj = root / "agreement" / name / "agreement.json"
        if aj.exists():
            inf = json.loads(aj.read_text()).get("informative")
            if inf:
                lines.append(f"- {name}: model labellers {json.dumps(inf, default=float)}")
        if bj.exists():
            lines.append(f"- {name}: corrected crossover by source {bj.read_text().strip()}")
    lines.append("")
    lines.append("## Part B")
    ft = root / "ft_SUMMARY.md"
    if ft.exists():
        lines.append(ft.read_text())
    lines.append("")
    lines.append("## Rating")
    rr = root / "rating" / "rating_report.json"
    if rr.exists():
        lines.append("```json"); lines.append(rr.read_text().strip()); lines.append("```")
    st = root / "rating" / "medgemma_status.json"
    if st.exists():
        lines.append(st.read_text())
    else:
        lines.append("MedGemma audit status was not written.")
    lines.append("")
    lines.append("The 240-image rating tool is `scripts/round6/rating_app.py` on 127.0.0.1:8765. "
                 "Intra-rater analysis runs only after both passes exist. "
                 "A4b is optional; if `a4b_KEY_SHA256.txt` is present and the author has not rated it, that is reported here.")
    if (root / "a4b_KEY_SHA256.txt").exists():
        lines.append(f"A4b key SHA-256: `{root.joinpath('a4b_KEY_SHA256.txt').read_text().strip()}`.")
        if not (root / "rating" / "a4b_pass1.csv").exists():
            lines.append("The author has not rated the A4b caliper sample.")
    text = "\n".join(lines) + "\n"
    (root / "SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
