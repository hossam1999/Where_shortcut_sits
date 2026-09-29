"""Round-9 exploratory search on development data only (docs/ROUND9_SEARCH_LEDGER.md). A SEARCH, not a test.

  python scripts/round9_search/search.py --candidate mask_ba [--smoke]     # fit on every development training set
  python scripts/round9_search/search.py --candidate mask_ba --evaluate    # development proxy criterion

Data (through firewall.check): development seeds 42/123/456/789/2026; traps thyroid, capsule, ISIC (their test
environments are development data); natural training sets of thyroid, capsule and the BCN / HAM / MSK hold-out designs,
of which only the train/validation part is ever built. Each natural validation fold is split by image group into a
selection half (C and the round-8 rule: worst-group objective, same-artifact guard) and an evaluation half (the D1'/D2'
proxy), so selection and evaluation never share images. Ovary, ISIC 2020 and every natural test set are never loaded.
Predictions: results/round9_search/preds/ (git-ignored); components and summary: results/round9_search/*.csv.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import json
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


FW = _load("r9_firewall", HERE / "firewall.py")
M = _load("r9_methods", HERE / "methods.py")
C = _load("round8_common", ROOT / "scripts" / "round8" / "common.py")
CA = _load("round8_candidates", ROOT / "scripts" / "round8" / "candidates.py")
HR = _load("round8_heads_run", ROOT / "scripts" / "round8" / "heads_run.py")
SB = _load("round8_scoreboard", ROOT / "scripts" / "round8" / "scoreboard.py")
R4 = C.load_round4_common()

from wtss import heads as H  # noqa: E402
from wtss.utils import stable_int  # noqa: E402

OUT = ROOT / "results" / "round9_search"
TRAP_TEST = ("test_rev", "test_corr", "clean")
N_BOOT = 2000


# ------------------------------------------------------------------------------------------------ data
def trainval_only(cohort: str):
    """Natural cohort table restricted to its train/validation part, feature arrays and index (test never built)."""
    rn = R4.load_script("rn_r9", "scripts/run_natural.py")
    c, _cache, fdir, _split = rn.load(cohort)
    if cohort == "thyroid":
        tv = c[c.split == "trainval"]
    elif cohort.startswith("isic_"):
        tv = c[c.source != cohort.split("_", 1)[1]]
    elif cohort == "capsule":
        # the capsule test split depends on the seed: built per seed in natural_sets (test groups dropped there)
        tv = c
    else:
        raise FW.FirewallError(cohort)
    V, pos = HR.load_features(fdir)
    return tv[tv.image_id.astype(str).isin(pos)].copy(), V, pos


def natural_sets(cohort: str, seeds, smoke: bool):
    FW.check("natural", cohort, seeds)
    tv0, V, pos = trainval_only(cohort)
    tv0["image_id"] = tv0.image_id.astype(str)
    sets = {}
    for s in seeds:
        tv = tv0
        if cohort == "capsule":  # drop this seed's test groups before anything else
            tv = tv0[[stable_int("natural_test", s, g) % 5 != 0 for g in tv0.group]]
        if smoke:
            tv = tv.sample(n=min(len(tv), 600), random_state=0)
        g = tv.group.unique()
        val_g = {x for x in g if stable_int("natural_val", s, x) % 5 == 0}
        va, tr = tv[tv.group.isin(val_g)], tv[~tv.group.isin(val_g)]
        half = np.array([stable_int("r9_val_half", s, x) % 2 == 0 for x in va.group])
        sel, ev = va[half], va[~half]
        cols = ["image_id", "y", "a"]
        E = {"train_corr": tr, "val_clean": sel, "val_groups": sel, "train_all": tr, "val_eval": ev}
        FW.check("natural", cohort, [s], E.keys())
        sets[("natural", s, 0)] = {k: d[cols].reset_index(drop=True) for k, d in E.items()}
    return sets, V, pos


def trap_sets(cohort: str, seeds, smoke: bool):
    FW.check("trap", cohort, seeds)
    sets, V, pos, _meta = HR.build("trap", cohort, "dino518", "dev", smoke)
    for k, E in sets.items():
        FW.check("trap", cohort, [k[1]], E.keys())
    return sets, V, pos


# ------------------------------------------------------------------------------------------------ candidates
def settings_for(name: str, ctx: dict) -> dict:
    """{setting: (view, head)} for one training set; the masking head is added by the caller as fallback."""
    Xm, Xe, y, a, Xmv, Xev, yv, seed = (ctx[k] for k in ("Xm", "Xe", "y", "a", "Xmv", "Xev", "yv", "seed"))
    fit = lambda X, Xv, **kw: M.select_C(lambda Cc: M.fit_linear(X, y, C=Cc, **kw), Xv, yv)[0]
    af = a.astype(float)
    out = {}
    if name == "mask_ba":
        out["ba"] = ("mask", fit(Xm, Xmv, free=af))
    elif name == "full_ba":
        out["ba"] = ("erm", fit(Xe, Xev, free=af))
    elif name == "mask_poe":
        out["poe"] = ("mask", fit(Xm, Xmv, offset=M.group_log_odds(y, a)))
    elif name == "mask_la":
        for tau in (0.5, 1.5, 2.0):
            out[f"tau{tau:g}"] = ("mask", fit(Xm, Xmv, offset=tau * M.group_log_odds(y, a)))
    elif name == "mask_vrex":
        g = M.groups(y, a)
        for bv in (1.0, 10.0, 100.0):
            out[f"beta{bv:g}"] = ("mask", fit(Xm, Xmv, penalty=M.vrex_penalty(g, bv)))
    elif name == "mask_irm":
        g = M.groups(y, a)
        for lam in (1.0, 10.0, 100.0):
            out[f"lam{lam:g}"] = ("mask", fit(Xm, Xmv, penalty=M.irm_penalty(g, y, lam)))
    elif name == "mask_moments":
        for gm in (1.0, 10.0, 100.0):
            out[f"gamma{gm:g}"] = ("mask", fit(Xm, Xmv, penalty=M.moment_penalty(y, a, gm)))
    elif name in ("mask_cmc_ba", "full_cmc_ba"):
        view, X, Xv = ("mask", Xm, Xmv) if name == "mask_cmc_ba" else ("erm", Xe, Xev)
        for mode in CA.CMC_MODES:
            P = CA.Projector(CA.cmc_directions(X, y, a, mode))
            out[mode] = (view, fit(X, Xv, free=af, transform=P))
    elif name == "mask_condadv":
        for lam in (0.1, 1.0, 10.0):
            P = M.train_conditional_adversary(Xm, y, a, lam, seed)
            out[f"lam{lam:g}"] = ("mask", CA.fit_weighted(Xm, y, CA.class_balanced_weights(y), Xmv, yv, seed, transform=P)[0])
    elif name == "mask_cnc":
        for wt in (0.5, 1.0, 2.0):
            P = M.train_cnc(Xm, y, a, wt, seed)
            out[f"w{wt:g}"] = ("mask", CA.fit_weighted(Xm, y, CA.class_balanced_weights(y), Xmv, yv, seed, transform=P)[0])
    elif name == "mask_cfc":
        Xcf, cf_index = ctx["cf"]
        for wt in (0.5, 1.0, 2.0):
            P = M.train_counterfactual_contrastive(Xm, y, Xcf, cf_index, wt, seed)
            out[f"w{wt:g}"] = ("mask", CA.fit_weighted(Xm, y, CA.class_balanced_weights(y), Xmv, yv, seed, transform=P)[0])
    elif name == "baselines":  # round-8 heads on development data, for comparison (not search candidates)
        pass
    else:
        raise ValueError(name)
    return out


_CTX: dict = {}


def run_set(job):
    key, E = job
    trap, seed, fold = key
    V, pos, name = _CTX["V"], _CTX["pos"], _CTX["name"]
    X = lambda v, d: V[v][[pos[i] for i in d.image_id.astype(str)]]
    tr, cv, vg = E["train_corr"], E["val_clean"], E["val_groups"]
    y, a, yv = tr.y.to_numpy().astype(int), tr.a.to_numpy().astype(int), cv.y.to_numpy().astype(int)
    ctx = dict(Xm=X("mask", tr), Xe=X("erm", tr), y=y, a=a, Xmv=X("mask", cv), Xev=X("erm", cv), yv=yv, seed=seed)
    if name == "mask_cfc":  # counterfactual renders: generic in-ROI overlay (every image) + real pasted instance (A0)
        ids = tr.image_id.astype(str).tolist()
        rows, src = [X("mask_insert", tr)], [np.arange(len(ids))]
        paste = _CTX.get("paste")
        if paste is not None:
            PX, pl = paste
            allowed = set(E["train_all"].image_id.astype(str)) | set(ids)
            pr, ps = [], []
            for j, i in enumerate(ids):
                if a[j] != 0:
                    continue
                for v in pl.get(i, []):
                    key_ = (v["k"], "in")
                    if v["donor"] in allowed and key_ in PX and i in PX[key_][1]:
                        pr.append(PX[key_][0][PX[key_][1][i]])
                        ps.append(j)
                        break
            if pr:
                rows.append(np.stack(pr))
                src.append(np.array(ps))
        ctx["cf"] = (np.vstack(rows), np.concatenate(src))
    mask_h = CA.Head(H.fit_erm(ctx["Xm"], y, ctx["Xmv"], yv, seed)[0])
    fam = {name: {"mask": ("mask", mask_h), **settings_for(name, ctx)}}
    if name == "baselines":
        fam = {"erm": {"erm": ("erm", CA.Head(H.fit_erm(ctx["Xe"], y, ctx["Xev"], yv, seed)[0]))},
               "mask_balanced": {"mask_balanced": ("mask", CA.fit_weighted(ctx["Xm"], y, CA.mask_bal_weights(y, a, 1.0),
                                                                          ctx["Xmv"], yv, seed)[0])},
               "mask_cmc": {"mask": ("mask", mask_h), **{m: ("mask", CA.fit_cmc(ctx["Xm"], y, a, ctx["Xmv"], yv, seed, m)[0])
                                                         for m in CA.CMC_MODES}},
               "full_cmc": {"mask": ("mask", mask_h), **{m: ("erm", CA.fit_cmc(ctx["Xe"], y, a, ctx["Xev"], yv, seed, m)[0])
                                                         for m in CA.CMC_MODES}},
               "mask_bal": {"mask": ("mask", mask_h), **{f"lam{l:g}": ("mask", CA.fit_weighted(
                   ctx["Xm"], y, CA.mask_bal_weights(y, a, l), ctx["Xmv"], yv, seed)[0]) for l in CA.LAMBDAS}}}
    yg, ag = vg.y.to_numpy().astype(int), vg.a.to_numpy().astype(int)
    rows, arms = [], {"mask": ("fixed", "mask", mask_h)}
    for f, settings in fam.items():
        if len(settings) == 1:
            (s, (v, h)), = settings.items()
            arms[f] = (s, v, h)
            continue
        val = {s: (yg, ag, h.prob(X(v, vg))) for s, (v, h) in settings.items()}
        choice, t = C.select_setting(val, reference="mask")
        rows += [{**r, "family": f, "trap": trap, "seed": seed, "fold": fold} for r in t.to_dict("records")]
        arms[f] = (choice, *settings[choice])
    frames = []
    envs = TRAP_TEST if trap != "natural" else ("val_eval",)
    for env in envs:
        d = E[env]
        for f, (s, v, h) in arms.items():
            frames.append(pd.DataFrame({"trap": trap, "seed": seed, "fold": fold, "method": f, "setting": s, "env": env,
                                        "image_id": d.image_id.astype(str).to_numpy(), "y": d.y.to_numpy(),
                                        "artifact_present": d.a.to_numpy(), "prob": h.prob(X(v, d))}))
    return rows, frames


def fit(name: str, smoke: bool, jobs: int):
    seeds = C.SMOKE_SEEDS if smoke else FW.ALLOWED_SEEDS
    if not smoke:
        FW.check("trap", "thyroid", seeds)
    base = OUT / ("_smoke" if smoke else "") / "preds"
    base.mkdir(parents=True, exist_ok=True)
    global _CTX
    choice_rows = []
    for kind, cohorts in (("trap", FW.TRAP_COHORTS), ("natural", FW.NATURAL_COHORTS)):
        for coh in cohorts:
            f = base / f"{name}__{kind}_{coh}.csv.gz"
            if f.exists():
                continue
            if smoke:  # smoke: the firewall accepts only development seeds, so check with one and use the smoke seed
                FW.check(kind, coh, [42])
            sets, V, pos = (trap_sets if kind == "trap" else natural_sets)(coh, seeds if not smoke else [42], smoke) \
                if not smoke else _smoke_sets(kind, coh)
            paste = None
            key = HR.LOCRAND_KEY.get(coh)
            if name == "mask_cfc" and key and (HR.LR.feat_dir(key, smoke) / "p0_in.npz").exists():
                paste = HR.LR.load_paste_features(key, smoke)
            _CTX = dict(V=V, pos=pos, name=name, paste=paste)
            js = sorted(sets.items(), key=lambda kv: kv[0])
            if smoke:
                js = [j for j in js if j[0][2] == 0][:2]
            res = R4.parallel_map(run_set, js) if jobs > 1 else [run_set(j) for j in js]
            p = pd.concat([fr for _, ff in res for fr in ff], ignore_index=True)
            p.to_csv(f, index=False, compression="gzip")
            choice_rows += [dict(r, kind=kind, cohort=coh) for rr, _ in res for r in rr]
            C.log(candidate=name, kind=kind, cohort=coh, rows=len(p))
    if choice_rows:
        pd.DataFrame(choice_rows).to_csv(OUT / ("_smoke" if smoke else "") / f"choices_{name}.csv", index=False,
                                         float_format="%.5f")


def _smoke_sets(kind, coh):
    """Smoke: development seed 42 only, small subsets (the firewall has already accepted seed 42)."""
    if kind == "trap":
        sets, V, pos, _ = HR.build("trap", coh, "dino518", "dev", True)
        for k, E in sets.items():
            FW.check("trap", coh, [k[1]], E.keys(), smoke=True)
        return sets, V, pos
    return natural_sets(coh, [42], True)


# ------------------------------------------------------------------------------------------------ evaluation
def _reps(job):
    label, p, kind, rule = job
    p = p[p.env.isin(TRAP_TEST if kind == "trap" else ("val_eval",))].copy()
    if kind == "trap":
        p["cell"] = p.trap
    else:
        hard = SB.natural_hard(p, rule)
        p = pd.concat([p.assign(cell="all"), p[hard].assign(cell="hard")])
    p = p.rename(columns={"method": "arm"})
    pt, rp = C.joint_replicates(C.terms_from_predictions(p, ["cell", "arm", "env"]), N_BOOT, C.stable_seed("r9", label))
    return label, {k: (pt[k], rp[k]) for k in pt}


def evaluate(name: str, smoke: bool) -> pd.DataFrame:
    A = _load("round8_analyse", ROOT / "scripts" / "round8" / "analyse.py")
    base = OUT / ("_smoke" if smoke else "") / "preds"
    jobs = []
    for kind, cohorts in (("trap", FW.TRAP_COHORTS), ("natural", FW.NATURAL_COHORTS)):
        for coh in cohorts:
            f = base / f"{name}__{kind}_{coh}.csv.gz"
            p = pd.read_csv(f)
            p["image_id"] = p.image_id.astype(str)
            jobs.append((f"{kind}/{coh}", p, kind, "prevalence"))
    S = dict(R4.parallel_map(_reps, jobs))
    arms = sorted({k[1] for R in S.values() for k in R} - {"mask"})
    rows = []
    for arm in arms:
        for coh in FW.NATURAL_COHORTS:
            c = A.contrast(S[f"natural/{coh}"], "all", arm, "val_eval")
            if c:
                rows.append({"arm": arm, "component": "D1' val all pairs", "cohort": coh, **A.summarise(*c, 0.01, "NI")})
        for coh in ("thyroid", "isic_BCN", "isic_MSK"):
            c = A.contrast(S[f"natural/{coh}"], "hard", arm, "val_eval")
            if c:
                rows.append({"arm": arm, "component": "D2' val hard pairs", "cohort": coh, **A.summarise(*c, 0.0, "SUP")})
        for coh in FW.TRAP_COHORTS:
            R = S[f"trap/{coh}"]
            c = A.contrast(R, "trapA", arm, "min_rev_corr")
            if c:
                rows.append({"arm": arm, "component": "D2' Trap A min(rev,corr)", "cohort": coh, **A.summarise(*c, 0.0, "SUP")})
            for cell, env, nm in A.D3_TYPES:
                c = A.contrast(R, cell, arm, env)
                if c:
                    rows.append({"arm": arm, "component": f"D3' {nm}", "cohort": coh, **A.summarise(*c, 0.03, "NI")})
            for trap in ("trapA", "trapB"):
                if (trap, arm, "test_rev") in R and (trap, arm, "test_corr") in R:
                    rev, corr = R[(trap, arm, "test_rev")][0], R[(trap, arm, "test_corr")][0]
                    ok = bool(corr >= rev - 0.02)
                    rows.append({"arm": arm, "component": "D4 no flipping", "cohort": f"{coh} {trap}", "estimate": corr - rev,
                                 "test": "point", "margin": 0.02, "met": ok, "label": "met" if ok else "loss"})
    comp = pd.DataFrame(rows)
    comp.insert(0, "candidate", name)
    out = OUT / ("_smoke" if smoke else "")
    comp.to_csv(out / f"components_{name}.csv", index=False, float_format="%.4f")
    summ = []
    for arm, g in comp.groupby("arm"):
        t = g[g.test.isin(["NI", "SUP"])]
        slack = (t.ci95_lo + np.where(t.test == "NI", t.margin, 0.0))  # lower bound minus the threshold
        worst = t.iloc[int(np.argmin(slack.to_numpy()))] if len(t) else None
        summ.append({"candidate": name, "arm": arm, "components": len(g), "met": int(g.met.sum()),
                     "losses": int((g.label == "loss").sum()), "inconclusive": int((g.label == "inconclusive").sum()),
                     "worst_slack": float(slack.min()) if len(t) else np.nan,
                     "worst_component": f"{worst.component} {worst.cohort} {worst.estimate:+.3f}" if worst is not None else ""})
    summ = pd.DataFrame(summ)
    stab = []
    for coh in FW.TRAP_COHORTS:
        p = next(j[1] for j in jobs if j[0] == f"trap/{coh}")
        q = p[(p.trap == "trapA") & (p.env == "test_rev")]
        for arm in arms:
            d = []
            for s, g in q.groupby("seed"):
                ga, gm = g[g.method == arm], g[g.method == "mask"]
                if len(ga) and len(gm):
                    d.append(C.all_pairs_auc(ga.y, ga.prob) - C.all_pairs_auc(gm.y, gm.prob))
            stab.append({"arm": arm, "cohort": coh, "sd": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan})
    st = pd.DataFrame(stab).groupby("arm").sd.mean().rename("trapA_rev_seed_sd").reset_index()
    summ = summ.merge(st, on="arm", how="left")
    f = out / "summary.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if len(old):
        old = old[old.candidate != name]
    pd.concat([old, summ], ignore_index=True).to_csv(f, index=False, float_format="%.4f")
    C.log(evaluated=name, **{r["arm"]: f"{r['met']}/{r['components']}" for r in summ.to_dict("records")})
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.evaluate:
        evaluate(a.candidate, a.smoke)
    else:
        fit(a.candidate, a.smoke, a.jobs)


if __name__ == "__main__":
    main()
