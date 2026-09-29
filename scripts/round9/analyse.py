"""Round 9 analysis (docs/PREREGISTRATION_ROUND9.md §5, §7).

  python scripts/round9/analyse.py [--smoke] [--n-boot 10000]

Primary: round-8 components (D1–D4) for mask_condadv, mask_irm, mask_vrex; intersection–union within a candidate;
fixed sequence in that order at one-sided 0.025 (all cohorts, and without ovary). Descriptive: the same components for
the round-8 references mask_cmc and mask_bal; ISIC 2019 -> 2020 all-pairs decomposition (easy / hard / same-artifact);
seed-level SD of the Trap A reversed contrast. Secondary: replication family R (Holm within).
Outputs: results/round9/{components,verdicts,replication,isic2020_decomposition,stability}.csv and SUMMARY.md.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


C = _load("round8_common", ROOT / "scripts" / "round8" / "common.py")
A = _load("round8_analyse", ROOT / "scripts" / "round8" / "analyse.py")
PD = _load("round8_pair_decomposition", ROOT / "scripts" / "round8" / "pair_decomposition.py")
SB = _load("round8_scoreboard", ROOT / "scripts" / "round8" / "scoreboard.py")
R4 = C.load_round4_common()

CANDIDATES = ("mask_condadv", "mask_irm", "mask_vrex")
REFERENCES = ("mask_cmc", "mask_bal")
REPLICATION = {"medsiglip448": ("thyroid", "capsule", "ovary"), "convnext384": ("thyroid", "capsule")}


def decomposition(path: Path) -> pd.DataFrame:
    """ISIC 2019 -> 2020: all-pairs contrast with masking = easy + hard + same-artifact parts (per seed, averaged)."""
    p = pd.read_csv(path)
    p = p[p.env == "clean"].copy()
    p["image_id"] = p.image_id.astype(str)
    hard = SB.natural_hard(p, "hair")
    mpos = int(p.loc[hard & (p.y == 1), "artifact_present"].iloc[0] == 0) if (hard & (p.y == 1)).any() else 1
    per = []
    for (arm, seed), q in p.groupby(["method", "seed"]):
        d = PD.decompose(q, mpos)
        tot = sum(n for _, n in d.values())
        per.append({"arm": arm, "seed": seed, "all": C.all_pairs_auc(q.y, q.prob),
                    **{f"auc_{k}": v[0] for k, v in d.items()}, **{f"pi_{k}": v[1] / tot for k, v in d.items()}})
    per = pd.DataFrame(per)
    m = per[per.arm == "mask"].set_index("seed")
    rows = []
    for arm, g in per.groupby("arm"):
        if arm == "mask":
            continue
        g = g.set_index("seed")
        r = {"arm": arm, "pi_easy": g.pi_easy.mean(), "pi_hard": g.pi_hard.mean(),
             "pi_same": (g.pi_same_a1 + g.pi_same_a0).mean()}
        parts = {}
        for k in ("easy", "hard", "same_a1", "same_a0"):
            parts[k] = float(np.nanmean(g[f"pi_{k}"] * (g[f"auc_{k}"] - m[f"auc_{k}"])))
        r.update({"easy_part": parts["easy"], "hard_part": parts["hard"],
                  "same_artifact_part": parts["same_a1"] + parts["same_a0"],
                  "all_pairs_delta": float((g["all"] - m["all"]).mean())})
        rows.append(r)
    return pd.DataFrame(rows)


def stability(path: Path, cohort: str) -> list:
    p = pd.read_csv(path)
    q = p[(p.trap == "trapA") & (p.env == "test_rev")]
    out = []
    for arm in sorted(set(q.method) - {"mask"}):
        d = []
        for s, g in q.groupby("seed"):
            ga, gm = g[g.method == arm], g[g.method == "mask"]
            if len(ga) and len(gm):
                d.append(C.all_pairs_auc(ga.y, ga.prob) - C.all_pairs_auc(gm.y, gm.prob))
        out.append({"cohort": cohort, "arm": arm, "trapA_rev_delta_mean": float(np.mean(d)) if d else np.nan,
                    "trapA_rev_delta_seed_sd": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    n_boot = 200 if a.smoke else a.n_boot
    out = ROOT / "results" / "round9" / ("_smoke" if a.smoke else "")
    conf = out / "confirm"
    jobs = []
    for coh in A.TRAPS:
        f = conf / "trap" / f"{coh}_dino518" / "predictions.csv.gz"
        if f.exists():
            jobs.append((f"trap/{coh}", f, "trap", None, n_boot))
    for coh in A.NATURAL:
        f = conf / "natural" / f"{coh}_dino518" / "predictions.csv.gz"
        if f.exists():
            jobs.append((f"natural/{coh}", f, "natural", "hair" if coh == "isic2020" else "prevalence", n_boot))
    for enc, cohs in REPLICATION.items():
        for coh in cohs:
            f = conf / "trap" / f"{coh}_{enc}" / "predictions.csv.gz"
            if f.exists():
                jobs.append((f"trap_{enc}/{coh}", f, "trap", None, n_boot))
    C.log(sources=len(jobs))
    S = dict(R4.parallel_map(A._source, jobs))

    comp_rows, verdicts = [], []
    for scope, inc in (("all cohorts", True), ("without ovary", False)):
        block = []
        for cand in CANDIDATES + REFERENCES:
            rows = A.primary_components(S, cand, inc)
            comp_rows += [{"scope": scope, "candidate": cand, **r} for r in rows]
            p, flips = A.iut(rows)
            d2 = [r for r in rows if r["component"].startswith("D2")]
            block.append({"scope": scope, "candidate": cand, "role": "candidate" if cand in CANDIDATES else "reference",
                          "p_iut": p, "no_flipping": flips, "components": len(rows),
                          "components_met": int(sum(bool(r["met"]) for r in rows)),
                          "losses": int(sum(r.get("label") == "loss" for r in rows)),
                          "all_D2_met": bool(d2) and all(r["met"] for r in d2)})
        cand_block = A.fixed_sequence([b for b in block if b["role"] == "candidate"])
        refs = [dict(b, sequence="descriptive reference", dominates=None) for b in block if b["role"] == "reference"]
        verdicts += cand_block + refs
    comp = pd.DataFrame(comp_rows)
    ver = pd.DataFrame(verdicts)

    rep = []
    for enc, cohs in REPLICATION.items():
        for coh in cohs:
            R = S.get(f"trap_{enc}/{coh}")
            for cand in CANDIDATES:
                c = A.contrast(R, "trapA", cand, "min_rev_corr") if R else None
                if c:
                    rep.append({"encoder": enc, "cohort": coh, "candidate": cand, **A.summarise(*c, 0.0, "SUP")})
    rep = pd.DataFrame(rep)
    if len(rep):
        rep["p_holm"] = C.holm(rep.p_one_sided.to_numpy())
        rep["met_holm"] = rep.p_holm <= A.ALPHA

    f20 = conf / "natural" / "isic2020_dino518" / "predictions.csv.gz"
    dec = decomposition(f20) if f20.exists() else pd.DataFrame()
    stab = []
    for coh in A.TRAPS:
        f = conf / "trap" / f"{coh}_dino518" / "predictions.csv.gz"
        if f.exists():
            stab += stability(f, coh)
    stab = pd.DataFrame(stab)

    comp.to_csv(out / "components.csv", index=False, float_format="%.4f")
    ver.to_csv(out / "verdicts.csv", index=False, float_format="%.4f")
    rep.to_csv(out / "replication.csv", index=False, float_format="%.4f")
    dec.to_csv(out / "isic2020_decomposition.csv", index=False, float_format="%.4f")
    stab.to_csv(out / "stability.csv", index=False, float_format="%.4f")
    write_summary(out / "SUMMARY.md", comp, ver, rep, dec, stab, a.smoke)
    C.log(written=str(out))


def decision(ver: pd.DataFrame) -> str:
    v = ver[(ver.scope == "all cohorts") & (ver.role == "candidate")]
    if v.dominates.fillna(False).astype(bool).any():
        return "dominates: " + ", ".join(v[v.dominates.fillna(False).astype(bool)].candidate)
    if v.all_D2_met.any():
        return "wins where masking fails but costs elsewhere: " + ", ".join(v[v.all_D2_met].candidate)
    return "no gain"


def write_summary(path, comp, ver, rep, dec, stab, smoke):
    lines = ["# Round 9 — do the search's candidates dominate ROI masking?", "",
             "Registration: `docs/PREREGISTRATION_ROUND9.md` (the last remedy round). Seeds 9101–9505. Every contrast is "
             "arm − masking with a crossed seed × image 95% interval; one-sided p; dominance = intersection–union of D1–D4; "
             "fixed sequence mask_condadv → mask_irm → mask_vrex (one-sided 0.025 each). mask_cmc and mask_bal (round 8) are "
             "descriptive references on the same data.", ""]
    if smoke:
        lines += ["**SMOKE RUN — validation images stand in for every test set; no number here is a result.**", ""]
    lines += [f"**Decision (registered rule, all cohorts): {decision(ver)}**", ""]
    for cand in CANDIDATES + REFERENCES:
        g = comp[(comp.scope == "all cohorts") & (comp.candidate == cand)]
        ls = "; ".join(f"{r.component} {r.cohort} {r.estimate:+.3f}" for r in g[g.label == "loss"].itertuples()) or "none"
        ic = "; ".join(f"{r.component} {r.cohort} {r.estimate:+.3f}" for r in g[g.label == "inconclusive"].itertuples()) or "none"
        lines.append(f"- **{cand}**: {int(g.met.sum())}/{len(g)} met. Losses: {ls}. Inconclusive: {ic}.")
    lines += ["", "## Verdicts", "", R4.md_table(ver.fillna(""))]
    if len(dec):
        lines += ["", "## ISIC 2019 → 2020: all-pairs contrast with masking, decomposed (π × Δ per pair type)", "",
                  R4.md_table(dec)]
    if len(stab):
        lines += ["", "## Seed stability (Trap A reversed, arm − masking)", "", R4.md_table(stab)]
    if len(rep):
        lines += ["", "## Replication family R (Trap A min(rev, corr), Holm within)", "", R4.md_table(rep.fillna(""))]
    for (scope, cand), g in comp.groupby(["scope", "candidate"], sort=False):
        t = g[["component", "cohort", "test", "margin", "estimate", "ci95_lo", "ci95_hi", "p_one_sided", "label"]]
        lines += ["", f"## {cand} — {scope}", "", R4.md_table(t.fillna(""))]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
