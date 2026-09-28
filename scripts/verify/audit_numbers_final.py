"""Number-tracing audit for the paper, the supplement and the six stage reports (docs/PREREGISTRATION_FINAL.md).

Rules
1. Every interval "[lo, hi]" must have both endpoints (±0.001) on one line of a result file under the regenerated
   folders (results/rerun_2026-09-28/, results/audit/) — i.e. produced by the corrected (crossed) bootstrap. An interval
   that only traces to archived results is an error: no interval of the old estimator may appear.
2. Every other number written with three decimals (e.g. 0.147, +0.253) must occur (±0.001) in some result file
   (archived or regenerated).
Integers, percentages and numbers with fewer decimals are design constants or counts and are not checked (listed in
the summary as unchecked).

  python scripts/verify/audit_numbers_final.py [--docs paper|stages|all]   exit code 1 if anything fails to trace
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
NEW_ROOTS = [RES / "rerun_2026-09-28", RES / "audit"]
NUM = re.compile(r"[-+−]?\d*\.\d+")
CI = re.compile(r"\[\s*\$?([-+−]?\d*\.\d+)\s*,\s*\$?\s*([-+−]?\d*\.\d+)\s*\$?\s*\]")
THREE = re.compile(r"(?<![\d.])[-+−]?\d+\.\d{3}(?!\d)")


def fnum(s):
    return float(s.replace("−", "-").replace("+", ""))


def result_files(roots):
    for r in roots:
        for f in list(r.rglob("*.csv")) + list(r.rglob("*.json")) + list(r.rglob("*.md")):
            if "bootstrap_correction" in f.parts or f.stat().st_size > 30_000_000:
                continue
            yield f


def lines_of(roots):
    out = []
    for f in result_files(roots):
        txt = f.read_text(errors="ignore")
        if f.suffix == ".json":
            txt = re.sub(r"\[\s*([^\]]*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", txt)
        for line in txt.splitlines():
            ns = set()
            for m in NUM.findall(line):
                try:
                    ns.add(round(fnum(m), 3))
                except ValueError:
                    pass
            if len(ns) >= 1:
                out.append(ns)
    return out


def docs(which):
    P = [ROOT / "paper" / "main.tex", ROOT / "paper" / "supplement.tex", ROOT / "paper" / "claim_checklist.tex"] + \
        sorted((ROOT / "paper" / "sections").glob("*.tex")) + sorted((ROOT / "paper" / "tables").glob("*.tex"))
    S = sorted((ROOT / "stages").rglob("*.tex"))
    return {"paper": P, "stages": S, "all": P + S}[which]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", default="all"); a = ap.parse_args()
    new_lines = lines_of(NEW_ROOTS)
    all_nums = set().union(*lines_of([RES])) if True else set()
    near = lambda v, ns: any(round(v + d, 3) in ns for d in (0, 0.001, -0.001))
    n_ci = n_num = 0
    bad = []
    for f in docs(a.docs):
        if not f.exists():
            continue
        s = f.read_text().replace("$", "")
        s = re.sub(r"(?m)%.*$", "", s)  # comments
        spans = []
        for m in CI.finditer(s):
            lo, hi = fnum(m.group(1)), fnum(m.group(2))
            n_ci += 1
            spans.append(m.span())
            if not any(near(lo, ns) and near(hi, ns) for ns in new_lines):
                bad.append(f"INTERVAL {f.relative_to(ROOT)}:{s[:m.start()].count(chr(10)) + 1} [{m.group(1)}, {m.group(2)}]")
        for m in THREE.finditer(s):
            if any(a0 <= m.start() < b0 for a0, b0 in spans):
                continue
            n_num += 1
            if not near(fnum(m.group(0)), all_nums):
                bad.append(f"NUMBER   {f.relative_to(ROOT)}:{s[:m.start()].count(chr(10)) + 1} {m.group(0)}")
    for b in bad:
        print(b)
    print(f"{n_ci} intervals and {n_num} three-decimal numbers checked in {a.docs}; {len(bad)} failed to trace")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
