"""Append one candidate's development-proxy result to docs/ROUND9_SEARCH_LEDGER.md (automatic, after every candidate)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
name, idx = sys.argv[1], sys.argv[2]
out = ROOT / "results" / "round9_search"
s = pd.read_csv(out / "summary.csv")
row = s[(s.candidate == name) & (s.arm == name)]
comp = pd.read_csv(out / f"components_{name}.csv")
c = comp[comp.arm == name]
ch = out / f"choices_{name}.csv"
fb = ""
if ch.exists():
    q = pd.read_csv(ch)
    q = q[(q.family == name) & (q.chosen == True)]  # noqa: E712
    if len(q):
        fb = f"; settings chosen: {q.setting.value_counts().to_dict()}"
lines = [f"", f"### {idx}. {name}"]
if row.empty:
    lines.append("- crashed or produced no predictions (see results/round9_search/logs).")
else:
    r = row.iloc[0]
    losses = c[c.label == "loss"]
    inc = c[c.label == "inconclusive"]
    fmt = lambda g: "; ".join(f"{x.component} {x.cohort} {x.estimate:+.3f} [{x.ci95_lo:+.3f}]" if pd.notna(x.ci95_lo)
                              else f"{x.component} {x.cohort} {x.estimate:+.3f}" for x in g.itertuples()) or "none"
    lines += [f"- development proxy: **{int(r.met)}/{int(r.components)} components met**, worst slack {r.worst_slack:+.3f} "
              f"({r.worst_component}); Trap A reversed seed SD {r.trapA_rev_seed_sd:.3f}{fb}",
              f"- losses: {fmt(losses)}", f"- inconclusive: {fmt(inc)}",
              f"- status: {'meets every proxy component' if int(r.met) == int(r.components) else 'does not meet every proxy component'}"]
p = ROOT / "docs" / "ROUND9_SEARCH_LEDGER.md"
p.write_text(p.read_text() + "\n".join(lines) + "\n")
print("\n".join(lines))
