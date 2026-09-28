"""Round 8, Phase 1: markdown tables for docs/ROUND8_DESIGN.md, generated from the committed CSVs (no hand-typed numbers).

    python scripts/round8/design_tables.py [--smoke]
Reads results/round8/{scoreboard_existing,criterion_existing,criterion_existing_summary,pair_decomposition}.csv and
writes results/round8/design_tables.md (sections marked <!-- T:name -->, pasted into the design document).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r8(name):
    """Load a round-8 module by path under a unique name (earlier rounds also have a module called `common`)."""
    import importlib.util
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


C = _r8("common")
CC = _r8("criterion_check")

def ci(r, est="estimate"):
    return f"{r[est]:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"


def absci(r):
    return f"{r['estimate']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def targets(sb: pd.DataFrame) -> pd.DataFrame:
    """Per criterion cell (DINOv2): masking's AUROC, the best existing arm and its contrast with masking."""
    rows = []
    for c in CC.cells():
        k = (sb.setting == c["setting"]) & (sb.cohort == c["cohort"]) & (sb.run == c["run"]) & (sb.cell == c["cell"]) \
            & (sb.env == c["env"])
        a = sb[k & (sb.metric == "auroc")]
        d = sb[k & (sb.metric == "delta_vs_mask")]
        m = a[a.arm == "mask"]
        if m.empty or d.empty:
            continue
        b = d.loc[d.estimate.idxmax()]
        rows.append({"component": c["component"], "cohort": c["cohort"], "type": c["kind"],
                     "mask AUROC [95% CI]": absci(m.iloc[0]), "best existing arm": b.arm,
                     "best − mask [95% CI]": ci(b)})
    return pd.DataFrame(rows)


def sweeps(sb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    s = sb[(sb.setting == "S1") & sb.run.str.endswith("corr_main") & (sb.env == "test_rev")]
    for (coh, enc, cell), g in s.groupby(["cohort", "encoder", "cell"]):
        m = g[(g.metric == "auroc") & (g.arm == "mask")]
        d = g[g.metric == "delta_vs_mask"]
        if m.empty or d.empty:
            continue
        b = d.loc[d.estimate.idxmax()]
        rows.append({"sweep": coh.replace("synthetic_", ""), "encoder": enc, "overlap": cell,
                     "mask reversed AUROC": absci(m.iloc[0]), "best arm": b.arm, "best − mask": ci(b)})
    return pd.DataFrame(rows)


def other_encoders(sb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    s = sb[(sb.setting == "S2") & (sb.encoder != "DINOv2-B/14@518") & (sb.env == "test_rev")]
    for (coh, enc, run, cell), g in s.groupby(["cohort", "encoder", "run", "cell"]):
        m = g[(g.metric == "auroc") & (g.arm == "mask")]
        d = g[g.metric == "delta_vs_mask"]
        if m.empty or d.empty:
            continue
        b = d.loc[d.estimate.idxmax()]
        rows.append({"cohort": coh, "encoder": enc, "run": run, "trap": cell, "mask reversed AUROC": absci(m.iloc[0]),
                     "best arm": b.arm, "best − mask": ci(b)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--doc", action="store_true", help="also fill the tables into docs/ROUND8_DESIGN.md")
    a = ap.parse_args()
    out = C.out_dir(a.smoke)
    sb = pd.read_csv(out / "scoreboard_existing.csv")
    cr = pd.read_csv(out / "criterion_existing.csv")
    cs = pd.read_csv(out / "criterion_existing_summary.csv")
    pdc = pd.read_csv(out / "pair_decomposition.csv")
    parts = []
    parts.append("<!-- T:targets -->\n" + md(targets(sb)))
    s = cs[["arm", "cells_run", "cells_met", "cells_total"]].copy()
    parts.append("<!-- T:criterion_summary -->\n" + md(s))
    keep = ["mask_balanced", "umte_balanced", "umte"]
    f = cr[cr.arm.isin(keep) & ~cr.met].copy()
    f["arm − mask [95% CI]"] = f.apply(ci, axis=1)
    parts.append("<!-- T:criterion_fails -->\n" + md(f[["arm", "component", "cohort", "kind", "arm − mask [95% CI]"]]))
    prec = cr.groupby("component").half_width.agg(["min", "median", "max"]).round(3).reset_index()
    prec.columns = ["component", "min half-width", "median half-width", "max half-width"]
    parts.append("<!-- T:precision -->\n" + md(prec))
    d = pdc[pdc.arm.isin(["mask_balanced", "umte_balanced", "balanced", "umte"])].copy()
    d = d[["cohort", "arm", "pi_easy", "pi_hard", "contrib_easy", "contrib_hard", "contrib_same", "d_all_observed"]]
    d = d.rename(columns={"pi_easy": "π easy", "pi_hard": "π hard", "contrib_easy": "easy part",
                          "contrib_hard": "hard part", "contrib_same": "same-artifact part", "d_all_observed": "Δ all pairs"})
    for c in d.columns[2:]:
        d[c] = d[c].map(lambda v: f"{v:+.3f}" if not c.startswith("π") else f"{v:.3f}")
    parts.append("<!-- T:decomposition -->\n" + md(d))
    parts.append("<!-- T:sweeps -->\n" + md(sweeps(sb)))
    parts.append("<!-- T:other_encoders -->\n" + md(other_encoders(sb)))
    corr = sb[(sb.metric == "delta_vs_mask") & (sb.env == "test_corr") & (sb.encoder == "DINOv2-B/14@518")
              & sb.run.str.contains("universal") & sb.arm.isin(["mask_balanced", "umte_balanced", "balanced"])].copy()
    corr["arm − mask [95% CI]"] = corr.apply(ci, axis=1)
    parts.append("<!-- T:corr_cost -->\n" + md(corr[["cohort", "cell", "arm", "arm − mask [95% CI]"]]
                                                 .sort_values(["cohort", "cell", "arm"])))
    (out / "design_tables.md").write_text("\n\n".join(parts) + "\n")
    C.log(written=str(out / "design_tables.md"))
    if a.doc and not a.smoke:
        insert(C.ROOT / "docs" / "ROUND8_DESIGN.md", parts)


def insert(doc: Path, parts: list[str]) -> None:
    """Replace each <!-- INSERT:name --> ... <!-- /INSERT:name --> block (or a bare marker) by the generated table."""
    import re
    text = doc.read_text()
    for p in parts:
        name = p.split("-->")[0].replace("<!-- T:", "").strip()
        body = p.split("-->", 1)[1].strip()
        block = f"<!-- INSERT:{name} -->\n{body}\n<!-- /INSERT:{name} -->"
        pat = re.compile(rf"<!-- INSERT:{name} -->(?:.*?<!-- /INSERT:{name} -->)?", re.S)
        if pat.search(text):
            text = pat.sub(lambda _m: block, text, count=1)
    doc.write_text(text)
    C.log(inserted=str(doc))


if __name__ == "__main__":
    main()
