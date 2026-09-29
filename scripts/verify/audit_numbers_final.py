"""Number-tracing audit for the paper, the supplement and the nine stage reports (docs/PREREGISTRATION_FINAL.md).

Rules
1. Every interval "[lo, hi]" must have both endpoints (±0.001) on one line of a result file under the regenerated
   folders (results/rerun_2026-09-28/, results/audit/, results/round4/, results/round5/, results/round6/, results/round7/) — i.e. produced by the corrected (crossed) bootstrap. An interval
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
NEW_ROOTS = [RES / "rerun_2026-09-28", RES / "audit", RES / "round4", RES / "round5", RES / "round6", RES / "round7", RES / "round8", RES / "round9", RES / "round9_search"]
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


def json_records(txt):
    """Number sets of every JSON object whose own scalar values include an interval (ci95_lo / ci95_hi)."""
    import json
    try:
        j = json.loads(txt)
    except ValueError:
        return []
    recs = []

    def walk(v):
        if isinstance(v, dict):
            sc = {round(float(x), 3) for k, x in v.items()
                  if isinstance(x, (int, float)) and not isinstance(x, bool) and not ARCHIVED.match(str(k))}
            if {"ci95_lo", "ci95_hi"} <= set(v) and sc:
                recs.append(sc)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(j)
    return recs


ARCHIVED = re.compile(r"^(orig|archived|old|exp|expected|pilot|ci_exp|delta_exp|clean_exp|rev_exp)", re.I)


def strict_lines(f):
    """Rows of a regenerated result file without the columns that carry archived / pilot reference values (e.g.
    orig_lo in review3/crossed_ci.csv, archived_lo in review2/R0_R1_crossovers.csv, the pilot's expected values in
    COMPARISON.csv and SUMMARY.md). Returns None for file types handled by the generic reader."""
    if f.suffix == ".csv":
        import pandas as pd
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            return []
        d = d[[c for c in d.columns if not ARCHIVED.match(str(c))]]
        out = []
        for row in d.astype(str).itertuples(index=False):
            ns = set()
            for cell in map(str, row):
                for m in NUM.findall(cell):
                    try:
                        ns.add(round(fnum(m), 3))
                    except ValueError:
                        pass
            if ns:
                out.append(ns)
        return out
    if f.suffix == ".md":
        out, drop = [], set()
        for line in f.read_text(errors="ignore").splitlines():
            if line.startswith("|"):
                cells = [c.strip() for c in re.split(r"\s\|\s", " " + line.strip().strip("|") + " ")]
                if any(ARCHIVED.match(c) for c in cells) and not any(NUM.search(c) for c in cells):
                    drop = {k for k, c in enumerate(cells) if ARCHIVED.match(c)}  # header row
                    continue
                line = " | ".join(c for k, c in enumerate(cells) if k not in drop)
            ns = set()
            for m in NUM.findall(line):
                try:
                    ns.add(round(fnum(m), 3))
                except ValueError:
                    pass
            if ns:
                out.append(ns)
        return out
    return None


BRACKET = re.compile(r"\[\s*([-+−]?\d*\.\d+)\s*,\s*([-+−]?\d*\.\d+)\s*\]")


def interval_pairs(roots):
    """Every (lo, hi) that a regenerated result file states as an interval: column pairs X_lo / X_hi (and lo / hi) of
    CSV rows, JSON objects with *_lo / *_hi keys or [estimate, lo, hi] / [lo, hi] lists, and "[lo, hi]" strings in
    Markdown or CSV cells. Archived / pilot reference columns are ignored."""
    import json
    import pandas as pd
    pairs = {}
    r3 = lambda x: round(float(x), 3)

    def add(lo, hi, ests=()):
        try:
            k = (r3(lo), r3(hi))
        except (TypeError, ValueError):
            return
        e = pairs.setdefault(k, set())
        for x in ests:
            try:
                e.add(r3(x))
            except (TypeError, ValueError):
                pass

    for f in result_files(roots):
        if f.suffix == ".csv":
            try:
                d = pd.read_csv(f)
            except Exception:  # noqa: BLE001
                continue
            cols = [c for c in d.columns if not ARCHIVED.match(str(c))]
            for c in cols:
                c = str(c)
                h = c[:-2] + "hi" if c.endswith("lo") else None
                if h and h in d.columns:
                    num = d[[x for x in cols if pd.api.types.is_numeric_dtype(d[x])]]
                    for (lo, hi), (_, row) in zip(zip(d[c], d[h]), num.iterrows()):
                        add(lo, hi, row.dropna().tolist())
                if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
                    for cell in d[c].dropna().astype(str):
                        for m in BRACKET.finditer(cell):
                            pre = NUM.findall(cell[:m.start()])
                            add(fnum(m.group(1)), fnum(m.group(2)), [fnum(pre[-1])] if pre else [])
        elif f.suffix == ".json":
            try:
                j = json.loads(f.read_text(errors="ignore"))
            except ValueError:
                continue

            def walk(v):
                if isinstance(v, dict):
                    for k, x in v.items():
                        if ARCHIVED.match(str(k)):
                            continue
                        if str(k).endswith("lo") and (str(k)[:-2] + "hi") in v:
                            add(x, v[str(k)[:-2] + "hi"], [y for kk, y in v.items() if isinstance(y, (int, float))
                                                           and not isinstance(y, bool) and not ARCHIVED.match(str(kk))])
                        walk(x)
                elif isinstance(v, list):
                    if len(v) in (2, 3) and all(isinstance(x, (int, float)) for x in v):
                        add(v[-2], v[-1], v[:1] if len(v) == 3 else [])
                    for x in v:
                        walk(x)
            walk(j)
        elif f.suffix == ".md":
            drop = set()
            for line in f.read_text(errors="ignore").splitlines():
                if line.startswith("|"):
                    cells = [c.strip() for c in re.split(r"\s\|\s", " " + line.strip().strip("|") + " ")]
                    if any(ARCHIVED.match(c) for c in cells) and not any(NUM.search(c) for c in cells):
                        drop = {k for k, c in enumerate(cells) if ARCHIVED.match(c)}
                        continue
                    line = " | ".join(c for k, c in enumerate(cells) if k not in drop)
                ln = line.replace("−", "-")
                for m in BRACKET.finditer(ln):
                    pre = NUM.findall(ln[max(0, m.start() - 12):m.start()])
                    add(fnum(m.group(1)), fnum(m.group(2)), [fnum(pre[-1])] if pre else [])
    return pairs


def lines_of(roots, strict=False):
    out = []
    for f in result_files(roots):
        if strict:
            sl = strict_lines(f)
            if sl is not None:
                out.extend(sl)
                continue
        txt = f.read_text(errors="ignore")
        if f.suffix == ".json":
            txt = re.sub(r"\[\s*([^\]]*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", txt)
            out.extend(json_records(txt))  # an estimate and its interval are one record even when pretty-printed
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


def input_tree(roots):
    """The .tex files a document actually contains: the roots and everything reached through \\input / \\include."""
    seen, todo = [], list(roots)
    while todo:
        f = todo.pop(0)
        if not f.exists() or f in seen:
            continue
        seen.append(f)
        txt = re.sub(r"(?m)(?<!\\)%.*$", "", f.read_text(errors="ignore"))
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", txt):
            g = (ROOT / "paper" / m.group(1))
            todo.append(g if g.suffix == ".tex" else g.with_suffix(".tex"))
    return seen


def docs(which):
    P = input_tree([ROOT / "paper" / "main.tex", ROOT / "paper" / "supplement.tex"])
    S = sorted((ROOT / "stages").rglob("*.tex"))
    return {"paper": P, "stages": S, "all": P + S}[which]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", default="all"); a = ap.parse_args()
    new_pairs = interval_pairs(NEW_ROOTS)
    all_nums = set().union(*lines_of([RES])) if True else set()
    near = lambda v, ns: any(round(v + d, 3) in ns for d in (0, 0.001, -0.001))
    n_ci = n_num = 0
    bad = []
    for f in docs(a.docs):
        if not f.exists():
            continue
        s = f.read_text().replace("$", "").replace("--", " \u2013 ")  # LaTeX ranges: 0.773--0.784 is not a minus sign
        s = re.sub(r"(?m)%.*$", "", s)  # comments
        spans = []
        for m in CI.finditer(s):
            lo, hi = fnum(m.group(1)), fnum(m.group(2))
            n_ci += 1
            spans.append(m.span())
            pre = NUM.findall(s[max(0, m.start() - 14):m.start()])
            est = fnum(pre[-1]) if pre else None
            ok = False
            for d1 in (0, 0.001, -0.001):
                for d2 in (0, 0.001, -0.001):
                    ests = new_pairs.get((round(lo + d1, 3), round(hi + d2, 3)))
                    if ests is not None and (est is None or any(round(est + d3, 3) in ests for d3 in (0, 0.001, -0.001))):
                        ok = True
            if not ok:
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
