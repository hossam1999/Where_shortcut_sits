"""A1 addendum (docs/PREREGISTRATION_FINAL.md): old versus new intervals for the four archived analyses that had no
per-image predictions, recomputed from the regenerated predictions in results/rerun_2026-09-28/ with the crossed
seed x image bootstrap (wtss.stats_crossed, 10,000 replicates).

  WTSS_BOOTSTRAP=crossed python scripts/verify/archived_rerun_compare.py [--only lama,capsule_tmpl,drain,devices]

Appends to results/bootstrap_correction/archived_old_vs_new.{csv,md} (one row per archived interval or point estimate:
old estimate and CI, new estimate and CI, verdict = CI excludes zero, verdict changed, |point difference| > 0.03) and
writes the recomputed contrasts next to the new predictions. Analyses whose predictions do not exist are listed as
not regenerated.
"""
from __future__ import annotations

import os

os.environ.setdefault("WTSS_BOOTSTRAP", "crossed")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wtss.stats_crossed import difference_of_deltas, hierarchical_paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results"
NEW = ROOT / "results" / "rerun_2026-09-28"
OUT = ROOT / "results" / "bootstrap_correction"
N_BOOT = 10000


def paired(preds: pd.DataFrame, a: str, b: str, env: str, seed: int) -> tuple:
    q = preds[preds.method.isin([a, b]) & (preds.env == env)]
    r = hierarchical_paired_bootstrap(q, a, b, env, N_BOOT, seed)
    return r["seed_delta_mean"], r["ci95_lo"], r["ci95_hi"]


def row(analysis, file, key, old, new, note=""):
    oe, ol, oh = (old + (np.nan, np.nan, np.nan))[:3] if old is not None else (np.nan, np.nan, np.nan)
    ne, nl, nh = new if new is not None else (np.nan, np.nan, np.nan)
    exc = lambda lo, hi: bool(np.isfinite(lo) and (lo > 0 or hi < 0))
    return {"analysis": analysis, "file": file, "key": key, "old_est": oe, "old_lo": ol, "old_hi": oh,
            "new_est": ne, "new_lo": nl, "new_hi": nh,
            "old_excludes_0": exc(ol, oh) if np.isfinite(ol) else None, "new_excludes_0": exc(nl, nh),
            "verdict_changed": (exc(ol, oh) != exc(nl, nh)) if np.isfinite(ol) else None,
            "est_diff": ne - oe if np.isfinite(oe) and np.isfinite(ne) else np.nan,
            "est_diff_over_003": bool(np.isfinite(oe) and np.isfinite(ne) and abs(ne - oe) > 0.03), "note": note}


def lama() -> list:
    """lama_comparison.json: LaMa run (arms erm = LaMa-inpainted, mask = LaMa then mask) paired with the universal run
    (mask, erm, U-MtE arms) on the same deterministic environments; Trap A, reversed test."""
    rows = []
    old = json.loads((OLD / "lama_comparison.json").read_text())
    # source of the paired arms, as in the archived comparison: thyroid U-MtE arms from the protect_generic run (the
    # archived universal run had no protected arm), ovary from the universal run; the thyroid comparison with the
    # regenerated universal run is added as a labelled extra row
    SRC = {"thyroid": "dino518_protect_generic", "ovary": "dino518_universal"}
    for coh in ("thyroid", "ovary"):
        fl, fu = NEW / coh / "dino518_lama" / "predictions.csv.gz", NEW / coh / SRC[coh] / "predictions.csv.gz"
        if not fl.exists():
            rows.append(row("LaMa", "lama_comparison.json", coh, None, None, "not regenerated: no predictions"))
            continue
        L = pd.read_csv(fl)
        L = L[L.trap == "trapA"].replace({"method": {"erm": "lama", "mask": "mask_lama"}})
        U = pd.read_csv(fu)
        U = U[(U.trap == "trapA") & U.method.isin(["erm", "mask", "mte_protect", "mte_balanced"])]
        P = pd.concat([L, U], ignore_index=True)
        rec = {}
        for k, (a, b) in {"lama-mask": ("lama", "mask"), "mask_lama-mask": ("mask_lama", "mask"),
                          "mte_protect-mask_lama": ("mte_protect", "mask_lama"),
                          "mte_balanced-mask_lama": ("mte_balanced", "mask_lama"), "lama-erm": ("lama", "erm")}.items():
            new = paired(P, a, b, "test_rev", 20260929)
            rec[k] = [round(x, 4) for x in new]
            rows.append(row("LaMa", "lama_comparison.json", f"{coh}|{k}", tuple(old.get(f"{coh}|{k}", [np.nan] * 3)), new))
        if coh == "thyroid":
            U2 = pd.read_csv(NEW / coh / "dino518_universal" / "predictions.csv.gz")
            U2 = U2[(U2.trap == "trapA") & U2.method.isin(["mask", "mte_protect"])]
            new = paired(pd.concat([L, U2], ignore_index=True), "mte_protect", "mask_lama", "test_rev", 20260929)
            rows.append(row("LaMa", "lama_comparison.json", "thyroid|mte_protect-mask_lama (universal run)",
                            tuple(old["thyroid|mte_protect-mask_lama"]), new,
                            "extra: protected arm of the regenerated universal run instead of protect_generic"))
        (NEW / coh / "dino518_lama" / "lama_comparison_crossed.json").write_text(json.dumps(rec, indent=1))
        rows += from_boot_vs_erm("LaMa", coh, "dino518_lama")
    return rows


def from_boot_vs_erm(analysis, coh_dir, run, sub=""):
    """Every row of an archived bootstrap_vs_erm.csv against the regenerated one (same trap / arm / env)."""
    rows = []
    fo, fn = OLD / coh_dir / run / sub / "bootstrap_vs_erm.csv", NEW / coh_dir / run / sub / "bootstrap_vs_erm.csv"
    if not fo.exists():
        return rows
    o = pd.read_csv(fo)
    n = pd.read_csv(fn) if fn.exists() else pd.DataFrame(columns=o.columns)
    keys = [c for c in ("trap", "arm", "env", "source") if c in o]
    for r in o.itertuples():
        m = n
        for k in keys:
            m = m[m[k] == getattr(r, k)] if k in m else m
        new = (m.seed_delta_mean.iloc[0], m.ci95_lo.iloc[0], m.ci95_hi.iloc[0]) if len(m) else None
        rows.append(row(analysis, str(fo.relative_to(OLD)), "|".join(str(getattr(r, k)) for k in keys) + "|-erm",
                        (r.seed_delta_mean, r.ci95_lo, r.ci95_hi), new, "" if new else "not regenerated"))
    return rows


def from_paired_deltas(analysis, rel_old, rel_new, seed=20260929):
    """Every row of an archived paired_deltas.csv recomputed from the regenerated predictions."""
    rows = []
    fo, fp = OLD / rel_old / "paired_deltas.csv", NEW / rel_new / "predictions.csv.gz"
    o = pd.read_csv(fo)
    if not fp.exists():
        return [row(analysis, str(fo.relative_to(OLD)), "all rows", None, None, "not regenerated: no predictions")]
    P = pd.read_csv(fp)
    out = []
    for r in o.itertuples():
        q = P[P.trap == r.trap] if "trap" in o and "trap" in P else P
        cluster = "seed"
        new = paired(q, r.arm, r.ref, r.env, seed) if {r.arm, r.ref} <= set(q.method) else None
        key = "|".join(str(x) for x in ([r.trap] if "trap" in o else []) + [r.arm, r.ref, r.env])
        rows.append(row(analysis, str(fo.relative_to(OLD)), key, (r.seed_delta_mean, r.ci95_lo, r.ci95_hi), new,
                        "" if new else "arm not in the regenerated run"))
        if new:
            out.append({"key": key, "estimate": new[0], "ci95_lo": new[1], "ci95_hi": new[2]})
    pd.DataFrame(out).to_csv(NEW / rel_new / "paired_deltas_crossed.csv", index=False)
    return rows


def capsule_tmpl():
    return (from_paired_deltas("capsule template U-MtE", "capsule/dino518_protect_tmpl", "capsule/dino518_protect_tmpl")
            + from_boot_vs_erm("capsule template U-MtE", "capsule", "dino518_protect_tmpl"))


def drain():
    rows = []
    for run in ("raddino518_universal", "dino518_universal"):
        rows += from_paired_deltas("chest drains", f"cxr_drain/{run}", f"cxr_drain/{run}")
    return rows


def devices():
    rows = []
    for enc in ("raddino518", "medsiglip448", "raddino518_devmatched"):
        base = OLD / "cxr_traps" / enc
        if not base.exists():
            continue
        for d in sorted(p.name for p in base.iterdir() if p.is_dir()):
            rows += from_boot_vs_erm("chest devices", "cxr_traps", enc, d)
            fo, fn = base / d / "X3_crossover.json", NEW / "cxr_traps" / enc / d / "X3_crossover.json"
            if fo.exists():
                o = json.loads(fo.read_text())
                n = json.loads(fn.read_text()) if fn.exists() else None
                g = lambda z: (z["seed_delta_mean"], z["ci95_lo"], z["ci95_hi"])
                rows.append(row("chest devices", str(fo.relative_to(OLD)), f"{enc}|{d}|crossover", g(o),
                                g(n) if n else None, "" if n else "not regenerated"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="lama,capsule_tmpl,drain,devices")
    a = ap.parse_args()
    fns = {"lama": lama, "capsule_tmpl": capsule_tmpl, "drain": drain, "devices": devices}
    rows = []
    for k in a.only.split(","):
        rows += fns[k]()
    new = pd.DataFrame(rows)
    f = OUT / "archived_old_vs_new.csv"
    if f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old[~old.analysis.isin(new.analysis.unique())], new], ignore_index=True)
    new.to_csv(f, index=False, float_format="%.4f")
    write_md(new)
    print(new.groupby("analysis").agg(rows=("key", "size"), verdict_changed=("verdict_changed", "sum"),
                                      diff_over_003=("est_diff_over_003", "sum")).to_string())


def write_md(d: pd.DataFrame):
    fmt = lambda e, l, h: "—" if not np.isfinite(e) else (f"{e:+.3f}" + (f" [{l:+.3f}, {h:+.3f}]" if np.isfinite(l) else ""))
    lines = ["# Archived analyses regenerated (A1 addendum, docs/PREREGISTRATION_FINAL.md)", "",
             "Old: archived runs without per-image predictions (per-seed bootstrap or point estimate). New: regenerated "
             "runs in results/rerun_2026-09-28/ with saved predictions, crossed seed x image bootstrap (10,000). "
             "Verdict = CI excludes zero.", ""]
    for an, g in d.groupby("analysis", sort=False):
        ok = g[g.new_est.notna()]
        lines += [f"## {an}", "", f"- rows: {len(g)}; regenerated: {len(ok)}; verdict changed: "
                  f"{int(g.verdict_changed.fillna(False).astype(bool).sum())}; point estimates differing by more than 0.03: "
                  f"{int(g['est_diff_over_003'].sum())}", "",
                  "| file | key | old | new | verdict changed | Δ > 0.03 | note |", "|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.file} | {r.key} | {fmt(r.old_est, r.old_lo, r.old_hi)} | {fmt(r.new_est, r.new_lo, r.new_hi)} "
                         f"| {'' if r.verdict_changed is None or (isinstance(r.verdict_changed, float) and np.isnan(r.verdict_changed)) else ('**yes**' if r.verdict_changed else 'no')} "
                         f"| {"**yes**" if r.est_diff_over_003 else ""} | {r.note if isinstance(r.note, str) else ''} |")
        lines.append("")
    (OUT / "archived_old_vs_new.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
