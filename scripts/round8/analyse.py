"""Round 8 confirmation analysis (docs/PREREGISTRATION_ROUND8.md, sections 7–9).

  python scripts/round8/analyse.py [--smoke] [--n-boot 10000]

Reads the confirmation predictions (git-ignored) and writes results/round8/confirm/{components,verdicts,replication,
descriptive_all_cells}.csv and results/round8/SUMMARY.md. Every contrast is candidate − masking on shared crossed
seed × image replicates (common.joint_replicates); one-sided bootstrap p-values; dominance = intersection–union test
(IUT: p = max over components; a D4 violation fails it); Holm over the four primary candidates at one-sided 0.025.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r8(name):
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


C = _r8("common")
SB = _r8("scoreboard")
R4 = C.load_round4_common()

ALPHA = 0.025
TRAPS = ("thyroid", "capsule", "ovary", "isic")
NATURAL = ("thyroid", "capsule", "isic_BCN", "isic_HAM", "isic_MSK", "isic2020")
HARD = ("thyroid", "isic_BCN", "isic_MSK", "isic2020")
D3_TYPES = (("trapB", "test_rev", "Trap B reversed"), ("trapB", "min_rev_corr", "Trap B min(rev,corr)"),
            ("trapA", "clean", "clean (Trap A models)"), ("trapB", "clean", "clean (Trap B models)"))
SWEEP_NAMES = ("thyroid", "capsule", "ovary", "isic2018")  # NIH chest tube: cohort table not on this machine
D5_CANDIDATES = ("mask_bal", "mask_cmc", "full_cmc")
REPLICATION = {"medsiglip448": ("thyroid", "capsule", "ovary"), "convnext384": ("thyroid", "capsule")}


def _source(job):
    """(label, predictions path, kind, hard-pair rule, n_boot) -> (label, {(cell, method, env): (point, replicates)})."""
    label, path, kind, rule, n_boot = job
    p = pd.read_csv(path)
    p["image_id"] = p.image_id.astype(str)
    p = p[p.env != "val_groups"]
    if kind == "trap":
        p = p.assign(cell=p.trap)
    elif kind == "sweep":
        p = p.assign(cell="r=" + p.overlap.map(lambda v: f"{v:g}"))
    else:
        p = p[p.env == "clean"]
        hard = SB.natural_hard(p, rule)
        p = pd.concat([p.assign(cell="all"), p[hard].assign(cell="hard"), p[~hard].assign(cell="easy")])
    p = p.rename(columns={"method": "arm"})
    point, reps = C.joint_replicates(C.terms_from_predictions(p, ["cell", "arm", "env"]), n_boot,
                                     C.stable_seed("round8_confirm", label))
    return label, {k: (point[k], reps[k]) for k in point}


def contrast(R: dict, cell: str, arm: str, env: str):
    """(estimate, replicates) of arm − mask; env 'min_rev_corr' uses min(reversed, correlated) of each arm."""
    def get(m):
        if env == "min_rev_corr":
            a, b = R.get((cell, m, "test_rev")), R.get((cell, m, "test_corr"))
            if a is None or b is None:
                return None
            return min(a[0], b[0]), np.fmin(a[1], b[1])
        return R.get((cell, m, env))
    x, m = get(arm), get("mask")
    if x is None or m is None:
        return None
    return x[0] - m[0], x[1] - m[1]


def summarise(est, arr, margin, kind):
    a = arr[np.isfinite(arr)]
    lo, hi = C.ci(a)
    thr = -margin if kind == "NI" else 0.0
    p = float(max((a <= thr).mean(), 1 / max(len(a), 1))) if len(a) else float("nan")
    return {"estimate": est, "ci95_lo": lo, "ci95_hi": hi, "margin": margin, "test": kind, "p_one_sided": p,
            "met": bool(np.isfinite(p) and p <= ALPHA)}


def primary_components(S: dict, cand: str, include_ovary: bool = True) -> list:
    rows = []
    for coh in NATURAL:
        R = S.get(f"natural/{coh}")
        c = contrast(R, "all", cand, "clean") if R else None
        if c:
            rows.append({"component": "D1 all pairs", "cohort": coh, **summarise(*c, 0.01, "NI")})
    for coh in HARD:
        R = S.get(f"natural/{coh}")
        c = contrast(R, "hard", cand, "clean") if R else None
        if c:
            rows.append({"component": "D2 hard pairs", "cohort": coh, **summarise(*c, 0.0, "SUP")})
    traps = [t for t in TRAPS if include_ovary or t != "ovary"]
    for coh in traps:
        R = S.get(f"trap/{coh}")
        c = contrast(R, "trapA", cand, "min_rev_corr") if R else None
        if c:
            rows.append({"component": "D2 Trap A min(rev,corr)", "cohort": coh, **summarise(*c, 0.0, "SUP")})
    for cell, env, name in D3_TYPES:
        arrs, ests = [], []
        for coh in traps:
            R = S.get(f"trap/{coh}")
            c = contrast(R, cell, cand, env) if R else None
            if c:
                rows.append({"component": f"D3 {name}", "cohort": coh, **summarise(*c, 0.03, "NI")})
                ests.append(c[0])
                arrs.append(c[1])
        if arrs:
            n = min(len(x) for x in arrs)
            pooled = np.nanmean(np.stack([x[:n] for x in arrs]), 0)
            rows.append({"component": f"D3 {name} (pooled)", "cohort": "+".join(traps),
                         **summarise(float(np.mean(ests)), pooled, 0.01, "NI")})
    for coh in traps:
        R = S.get(f"trap/{coh}")
        for trap in ("trapA", "trapB"):
            if R and (trap, cand, "test_rev") in R and (trap, cand, "test_corr") in R:
                rev, corr = R[(trap, cand, "test_rev")][0], R[(trap, cand, "test_corr")][0]
                rows.append({"component": "D4 no flipping", "cohort": f"{coh} {trap}", "estimate": corr - rev,
                             "test": "point", "margin": 0.02, "met": bool(corr >= rev - 0.02), "p_one_sided": np.nan})
    return rows


def iut(rows: list) -> tuple:
    ps = [r["p_one_sided"] for r in rows if r["test"] in ("NI", "SUP")]
    flips_ok = all(r["met"] for r in rows if r["test"] == "point")
    if not ps:
        return float("nan"), False
    return (float(np.nanmax(ps)) if flips_ok else 1.0), flips_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    n_boot = 200 if a.smoke else a.n_boot
    conf = C.stage_dir("confirm", a.smoke)
    jobs = []
    for coh in TRAPS:
        f = conf / "trap" / f"{coh}_dino518" / "predictions.csv.gz"
        if f.exists():
            jobs.append((f"trap/{coh}", f, "trap", None, n_boot))
    for coh in NATURAL:
        f = conf / "natural" / f"{coh}_dino518" / "predictions.csv.gz"
        if f.exists():
            jobs.append((f"natural/{coh}", f, "natural", "hair" if coh == "isic2020" else "prevalence", n_boot))
    for enc, cohs in REPLICATION.items():
        for coh in cohs:
            f = conf / "trap" / f"{coh}_{enc}" / "predictions.csv.gz"
            if f.exists():
                jobs.append((f"trap_{enc}/{coh}", f, "trap", None, n_boot))
    for sw in SWEEP_NAMES:
        f = conf / "sweeps" / sw / "predictions.csv.gz"
        if f.exists():
            jobs.append((f"sweep/{sw}", f, "sweep", None, n_boot))
    for part in ("natural", "traps"):
        f = conf / "ft" / part / "predictions.csv.gz"
        if f.exists():
            jobs.append((f"ft/{part}", f, "trap" if part == "traps" else "natural", "caliper", n_boot))
    C.log(sources=len(jobs))
    S = dict(R4.parallel_map(_source, jobs))

    comp_rows, verdicts = [], []
    for scope, inc in (("all cohorts", True), ("without ovary", False)):
        block = []
        for cand in C.PRIMARY:
            rows = primary_components(S, cand, inc)
            comp_rows += [{"scope": scope, "candidate": cand, **r} for r in rows]
            p, flips = iut(rows)
            d2 = [r for r in rows if r["component"].startswith("D2")]
            block.append({"scope": scope, "candidate": cand, "p_iut": p, "no_flipping": flips, "components": len(rows),
                          "components_met": int(sum(bool(r["met"]) for r in rows)),
                          "all_D2_met": bool(d2) and all(r["met"] for r in d2)})
        adj = C.holm(np.nan_to_num([b["p_iut"] for b in block], nan=1.0))
        for b, q in zip(block, adj):
            b["p_holm"], b["dominates"] = float(q), bool(q <= ALPHA)
        verdicts += block
    comp = pd.DataFrame(comp_rows)
    ver = pd.DataFrame(verdicts)

    extra = []
    for cand in D5_CANDIDATES:
        rows = []
        for sw in SWEEP_NAMES:
            R = S.get(f"sweep/{sw}")
            if not R:
                continue
            for cell, env, kind, m in (("r=1", "test_rev", "SUP", 0.0), ("r=0", "test_rev", "NI", 0.03),
                                       ("r=0", "clean", "NI", 0.03)):
                c = contrast(R, cell, cand, env)
                if c:
                    rows.append({"component": f"D5 {cell} {env}", "cohort": sw, **summarise(*c, m, kind)})
        if rows:
            comp = pd.concat([comp, pd.DataFrame([{"scope": "D5", "candidate": cand, **r} for r in rows])],
                             ignore_index=True)
            p, _ = iut(rows)
            extra.append({"scope": "D5", "candidate": cand, "p_iut": p, "components": len(rows),
                          "components_met": int(sum(r["met"] for r in rows))})
    if extra:
        for v, q in zip(extra, C.holm(np.nan_to_num([v["p_iut"] for v in extra], nan=1.0))):
            v["p_holm"], v["dominates"] = float(q), bool(q <= ALPHA)
    d6 = []
    Rn, Rt = S.get("ft/natural"), S.get("ft/traps")
    for R, cell, env, kind, m, name in ((Rn, "all", "clean", "NI", 0.02, "D6 all pairs"),
                                        (Rn, "hard", "clean", "SUP", 0.0, "D6 hard pairs"),
                                        (Rt, "trapA", "min_rev_corr", "SUP", 0.0, "D6 Trap A min(rev,corr)")):
        c = contrast(R, cell, "locrand_ft", env) if R else None
        if c:
            d6.append({"component": name, "cohort": "thyroid (fine-tuned)", **summarise(*c, m, kind)})
    if d6:
        comp = pd.concat([comp, pd.DataFrame([{"scope": "D6", "candidate": "locrand_ft", **r} for r in d6])],
                         ignore_index=True)
        p, _ = iut(d6)
        extra.append({"scope": "D6", "candidate": "locrand_ft", "p_iut": p, "p_holm": p, "dominates": bool(p <= ALPHA),
                      "components": len(d6), "components_met": int(sum(r["met"] for r in d6))})
    ver = pd.concat([ver, pd.DataFrame(extra)], ignore_index=True)

    rep = []
    for enc, cohs in REPLICATION.items():
        for coh in cohs:
            R = S.get(f"trap_{enc}/{coh}")
            for cand in D5_CANDIDATES:
                c = contrast(R, "trapA", cand, "min_rev_corr") if R else None
                if c:
                    rep.append({"encoder": enc, "cohort": coh, "candidate": cand, **summarise(*c, 0.0, "SUP")})
    rep = pd.DataFrame(rep)
    if len(rep):
        rep["p_holm"] = C.holm(rep.p_one_sided.to_numpy())
        rep["met_holm"] = rep.p_holm <= ALPHA

    desc = []
    for label, R in S.items():
        keys = sorted({(k[0], k[2]) for k in R})
        arms = sorted({k[1] for k in R})
        cells = sorted({k[0] for k in R})
        envs = {e for _, e in keys}
        more = [(c, "min_rev_corr") for c in cells] if {"test_rev", "test_corr"} <= envs else []
        for cell, env in keys + more:
            for arm in arms:
                if arm == "mask":
                    continue
                c = contrast(R, cell, arm, env)
                if c:
                    lo, hi = C.ci(c[1])
                    desc.append({"source": label, "cell": cell, "env": env, "arm": arm, "arm_minus_mask": c[0],
                                 "ci95_lo": lo, "ci95_hi": hi})
    desc = pd.DataFrame(desc)

    comp.to_csv(conf / "components.csv", index=False, float_format="%.4f")
    ver.to_csv(conf / "verdicts.csv", index=False, float_format="%.4f")
    rep.to_csv(conf / "replication.csv", index=False, float_format="%.4f")
    desc.to_csv(conf / "descriptive_all_cells.csv", index=False, float_format="%.4f")
    write_summary(C.out_dir(a.smoke) / "SUMMARY.md", comp, ver, rep, a.smoke)
    C.log(written=str(conf), verdicts=len(ver))


def decision(ver: pd.DataFrame) -> str:
    v = ver[ver.scope == "all cohorts"]
    if v.empty:
        return "not available"
    if v.dominates.any():
        return "dominates: " + ", ".join(v[v.dominates].candidate)
    if v.all_D2_met.any():
        return "wins where masking fails but costs elsewhere: " + ", ".join(v[v.all_D2_met].candidate)
    return "no gain"


def write_summary(path: Path, comp: pd.DataFrame, ver: pd.DataFrame, rep: pd.DataFrame, smoke: bool):
    lines = ["# Round 8 — does any remedy dominate ROI masking?", "",
             "Registration: `docs/PREREGISTRATION_ROUND8.md`. Frozen development choices: "
             "`results/round8/frozen_choice.json`. Every contrast is candidate − masking with a crossed seed × image 95% "
             "interval; one-sided p; dominance = intersection–union of D1–D4; Holm over the four primary candidates "
             "(one-sided 0.025).", ""]
    if smoke:
        lines += ["**SMOKE RUN — validation images stand in for every test set; no number here is a result.**", ""]
    lines += [f"**Decision (registered rule, all cohorts): {decision(ver)}**", "", "## Verdicts", "",
              R4.md_table(ver.fillna(""))]
    for (scope, cand), g in comp.groupby(["scope", "candidate"], sort=False):
        t = g[["component", "cohort", "test", "margin", "estimate", "ci95_lo", "ci95_hi", "p_one_sided", "met"]]
        fails = g[~g.met.astype(bool)]
        lines += ["", f"## {cand} — {scope}", "", R4.md_table(t.fillna("")), "",
                  "Fails: " + ("none" if fails.empty else "; ".join(
                      f"{r.component} {r.cohort} {r.estimate:+.3f}" for r in fails.itertuples()))]
    if len(rep):
        lines += ["", "## Replication (MedSigLIP, ConvNeXt; Trap A min(rev,corr), Holm within)", "",
                  R4.md_table(rep.fillna(""))]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
