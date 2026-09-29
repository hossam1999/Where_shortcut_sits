"""Round 9 confirmation (docs/PREREGISTRATION_ROUND9.md).

  python scripts/round9/confirm.py --kind trap    --cohort thyroid [--encoder dino518] [--smoke]
  python scripts/round9/confirm.py --kind natural --cohort isic2020 [--smoke]

Training sets: scripts/round8/heads_run.build with the round-9 seeds 9101–9505 (smoke: seed 99991, small subsets,
validation images stand in for every test set). Arms: masking (reference), ERM, the three candidates (mask_condadv,
mask_irm, mask_vrex; heads of scripts/round9_search/methods.py, grids frozen) and the round-8 references mask_cmc and
mask_bal with their round-8 recipes. Selection per training set by the round-8 rule on val_groups; the held-out ovary
uses the frozen settings. Refuses to run (except smoke) unless docs/PREREGISTRATION_ROUND9.md is committed and unchanged.
Outputs: results/round9/confirm/<kind>/<cohort>_<encoder>/{choices.csv, predictions.csv.gz (git-ignored)}.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import subprocess
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
CA = _load("round8_candidates", ROOT / "scripts" / "round8" / "candidates.py")
HR = _load("round8_heads_run", ROOT / "scripts" / "round8" / "heads_run.py")
SR = _load("r9_search", ROOT / "scripts" / "round9_search" / "search.py")
R4 = C.load_round4_common()

from wtss import heads as H  # noqa: E402

R9_SEEDS = (9101, 9202, 9303, 9404, 9505)
CANDIDATES = ("mask_condadv", "mask_irm", "mask_vrex")          # fixed-sequence order
REFERENCES = ("mask_cmc", "mask_bal")                           # round 8, descriptive
HELD_OUT = "ovary"
HELD_OUT_SETTINGS = {"mask_condadv": "lam1", "mask_irm": "lam100", "mask_vrex": "beta100",   # round-9 development
                     "mask_cmc": "per_class", "mask_bal": "lam1"}                           # round-8 frozen_choice.json
OUT = ROOT / "results" / "round9"


def registered() -> bool:
    rel = "docs/PREREGISTRATION_ROUND9.md"
    tracked = subprocess.call(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    clean = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]) == 0
    return tracked and clean


_CTX: dict = {}


def fit_set(job):
    key, E = job
    trap, seed, fold = key
    V, pos, cohort, smoke = _CTX["V"], _CTX["pos"], _CTX["cohort"], _CTX["smoke"]
    X = lambda v, d: V[v][[pos[i] for i in d.image_id.astype(str)]]
    test_envs = [e for e in (("test_rev", "test_corr", "clean") if trap != "natural" else ("clean",)) if e in E]
    if smoke:
        E = C.smoke_envs_from_validation(E, test_envs)
    tr, cv, vg = E["train_corr"], E["val_clean"], E["val_groups"]
    y, a, yv = tr.y.to_numpy().astype(int), tr.a.to_numpy().astype(int), cv.y.to_numpy().astype(int)
    ctx = dict(Xm=X("mask", tr), Xe=X("erm", tr), y=y, a=a, Xmv=X("mask", cv), Xev=X("erm", cv), yv=yv, seed=seed)
    mask_h = CA.Head(H.fit_erm(ctx["Xm"], y, ctx["Xmv"], yv, seed)[0])
    fam = {c: {"mask": ("mask", mask_h), **SR.settings_for(c, ctx)} for c in CANDIDATES}
    fam["mask_cmc"] = {"mask": ("mask", mask_h), **{m: ("mask", CA.fit_cmc(ctx["Xm"], y, a, ctx["Xmv"], yv, seed, m)[0])
                                                    for m in CA.CMC_MODES}}
    fam["mask_bal"] = {"mask": ("mask", mask_h), **{f"lam{l:g}": ("mask", CA.fit_weighted(
        ctx["Xm"], y, CA.mask_bal_weights(y, a, l), ctx["Xmv"], yv, seed)[0]) for l in CA.LAMBDAS}}
    arms = {"mask": ("fixed", "mask", mask_h),
            "erm": ("fixed", "erm", CA.Head(H.fit_erm(ctx["Xe"], y, ctx["Xev"], yv, seed)[0]))}
    yg, ag = vg.y.to_numpy().astype(int), vg.a.to_numpy().astype(int)
    rows = []
    held = cohort == HELD_OUT and trap != "natural"
    for f, settings in fam.items():
        if held:
            choice = HELD_OUT_SETTINGS[f] if HELD_OUT_SETTINGS[f] in settings else "mask"
            rows.append({"family": f, "setting": choice, "chosen": True, "frozen": True})
        else:
            choice, t = C.select_setting({s: (yg, ag, h.prob(X(v, vg))) for s, (v, h) in settings.items()},
                                         reference="mask")
            rows += [{**r, "family": f, "frozen": False} for r in t.to_dict("records")]
        arms[f] = (choice, *settings[choice])
    for r in rows:
        r.update({"trap": trap, "seed": seed, "fold": fold})
    frames = []
    for env in ["val_groups"] + test_envs:
        d = E[env]
        cache = {}
        for f, (s, v, h) in arms.items():
            if v not in cache:
                cache[v] = X(v, d)
            frames.append(pd.DataFrame({"trap": trap, "seed": seed, "fold": fold, "method": f, "setting": s, "env": env,
                                        "image_id": d.image_id.astype(str).to_numpy(), "y": d.y.to_numpy(),
                                        "artifact_present": d.a.to_numpy(), "prob": h.prob(cache[v])}))
    return rows, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("trap", "natural"))
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--encoder", default="dino518", choices=tuple(HR.BDIR))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if not a.smoke and not registered():
        raise SystemExit("docs/PREREGISTRATION_ROUND9.md must be committed and unchanged before any round-9 data")
    # round-9 seeds (smoke: the smoke seed only) for heads_run.build
    C.seeds_for = lambda stage, smoke: C.SMOKE_SEEDS if smoke else R9_SEEDS
    out = OUT / ("_smoke" if a.smoke else "") / "confirm" / a.kind / f"{a.cohort}_{a.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predictions.csv.gz").exists():
        C.log(skip=str(out))
        return
    sets, V, pos, _meta = HR.build(a.kind, a.cohort, a.encoder, "confirm", a.smoke)
    seeds = sorted({k[1] for k in sets})
    assert seeds == sorted(C.SMOKE_SEEDS if a.smoke else R9_SEEDS), seeds
    global _CTX
    _CTX = dict(V=V, pos=pos, cohort=a.cohort, smoke=a.smoke)
    jobs = sorted(sets.items(), key=lambda kv: kv[0])
    if a.smoke:
        jobs = [j for j in jobs if j[0][2] == 0][:2]
    C.log(kind=a.kind, cohort=a.cohort, encoder=a.encoder, training_sets=len(jobs), seeds=seeds)
    res = R4.parallel_map(fit_set, jobs) if a.jobs > 1 else [fit_set(j) for j in jobs]
    ch = pd.DataFrame([r for rr, _ in res for r in rr])
    ch.insert(0, "cohort", a.cohort)
    ch.insert(1, "encoder", a.encoder)
    ch.to_csv(out / "choices.csv", index=False, float_format="%.5f")
    p = pd.concat([f for _, ff in res for f in ff], ignore_index=True)
    p.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    C.log(written=str(out), rows=len(p))


if __name__ == "__main__":
    main()
