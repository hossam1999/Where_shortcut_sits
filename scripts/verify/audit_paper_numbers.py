"""Trace every confidence interval in the paper to a machine-written result file.

Collects all numbers (3-decimal rounding) from results/**/*.csv|json|md (script outputs) and checks, for every
"[lo, hi]" interval in paper/*.tex and paper/sections/*.tex, that both endpoints occur (±0.0015). Unmatched intervals
are listed for manual review (they may be derived, e.g. differences reported in docs only).
  python scripts/verify/audit_paper_numbers.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from wtss import paths

NUM = re.compile(r"[-+−]?\d*\.\d+")
CI = re.compile(r"\[\s*\$?([-+−]?\d*\.\d+)\s*,\s*\$?\s*([-+−]?\d*\.\d+)\s*\$?\s*\]")


def fnum(s):
    return float(s.replace("−", "-").replace("+", ""))


def result_lines():
    """Per line of every result file: the set of its numbers (3 decimals). An interval is traced only if both
    endpoints occur on the same line (strict mode)."""
    out = []
    for f in list(paths.RESULTS.rglob("*.csv")) + list(paths.RESULTS.rglob("*.json")) + list(paths.RESULTS.rglob("*.md")):
        if f.stat().st_size > 20_000_000:
            continue
        txt = f.read_text(errors="ignore")
        if f.suffix == ".json":  # json intervals are often split across lines: collapse lists
            txt = re.sub(r"\[\s*([^\]]*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", txt)
        for line in txt.splitlines():
            ns = set()
            for m in NUM.findall(line):
                try:
                    ns.add(round(fnum(m), 3))
                except ValueError:
                    pass
            if len(ns) >= 2:
                out.append((ns, f))
    return out


def result_numbers():
    vals = set()
    for f in list(paths.RESULTS.rglob("*.csv")) + list(paths.RESULTS.rglob("*.json")) + list(paths.RESULTS.rglob("*.md")):
        if f.stat().st_size > 20_000_000:
            continue
        for m in NUM.findall(f.read_text(errors="ignore")):
            try:
                vals.add(round(fnum(m), 3))
            except ValueError:
                pass
    return vals


def main():
    lines = result_lines()
    near = lambda v, ns: any(round(v + d, 3) in ns for d in (0, 0.001, -0.001))
    has_pair = lambda lo, hi: any(near(lo, ns) and near(hi, ns) for ns, _ in lines)
    tex = sorted(set(Path(paths.REPO_ROOT / "paper").glob("*.tex")) | set(Path(paths.REPO_ROOT / "paper" / "sections").glob("*.tex")))
    n = bad = 0
    for f in tex:
        s = f.read_text().replace("$", "")
        for m in CI.finditer(s):
            lo, hi = fnum(m.group(1)), fnum(m.group(2))
            n += 1
            if not has_pair(lo, hi):
                bad += 1
                line = s[:m.start()].count("\n") + 1
                print(f"UNMATCHED {f.name}:{line}  [{m.group(1)}, {m.group(2)}]  …{s[max(0, m.start() - 80):m.start()].strip()[-80:]}")
    print(f"{n} intervals checked, {n - bad} traced (both endpoints on one line of a result file), {bad} to review")


if __name__ == "__main__":
    main()
