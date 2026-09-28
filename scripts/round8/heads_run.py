"""Round 8 runner for every frozen-feature candidate (docs/PREREGISTRATION_ROUND8.md, sections 3–5).

  python scripts/round8/heads_run.py --stage dev     --kind trap    --cohort thyroid [--encoder dino518]
  python scripts/round8/heads_run.py --stage confirm --kind trap    --cohort ovary
  python scripts/round8/heads_run.py --stage confirm --kind natural --cohort isic_BCN | thyroid | capsule | isic2020
  (--smoke: one smoke seed, small subsets; every "test" environment is replaced by the run's own validation split)

dev      development seeds (the environments of rounds 1–7); fits every candidate, applies the selection rule on the
         selection split, writes the choices and validation statistics only. No test environment is predicted.
         The held-out cohort (ovary) is never run in dev.
confirm  confirmation seeds; requires results/round8/frozen_choice.json to be committed; predicts the test
         environments with the selected setting of every candidate (the held-out cohort uses the frozen settings).
Outputs: <stage>/<kind>/<cohort>_<encoder>/choices.csv (committed) and predictions.csv.gz (git-ignored).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import json
import subprocess
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
CA = _r8("candidates")
LR = _r8("locrand")
R4 = C.load_round4_common()

from wtss import heads as H  # noqa: E402
from wtss import paths  # noqa: E402
from wtss.utils import stable_int  # noqa: E402

BDIR = {"dino518": "dinov2_b14_518", "medsiglip448": "medsiglip_448", "convnext384": "convnext_b_384"}
TRAP_FEAT = {"thyroid": "thyroid", "capsule": "capsule", "ovary": "ovary", "isic": "spec_isic2019"}
LOCRAND_KEY = {"thyroid": "thyroid", "capsule": "capsule", "ovary": "ovary", "isic": "isic2019",
               "isic_BCN": "isic2019", "isic_HAM": "isic2019", "isic_MSK": "isic2019", "isic2020": "isic2019"}
HELD_OUT = "ovary"
TRAP_TEST_ENVS = ("test_rev", "test_corr", "clean")
FIXED_DESCRIPTIVE = ("erm", "mask", "balanced", "mask_balanced", "umte", "umte_protect", "umte_balanced", "mask_dfr",
                     "groupdro")


# ------------------------------------------------------------------------------------------------ data
def _nisic():
    """round-4 natural_isic2020 (its `import common` must resolve to round-4 common)."""
    saved = sys.modules.get("common")
    sys.modules["common"] = R4
    sys.path.insert(0, str(C.ROOT / "scripts" / "round4"))
    spec = importlib.util.spec_from_file_location("natural_isic2020_r8", C.ROOT / "scripts" / "round4" / "natural_isic2020.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if saved is not None:
        sys.modules["common"] = saved
    else:
        sys.modules.pop("common", None)
    return mod


def load_features(fdir: Path, fname=lambda v: f"{v}.npz", views=("erm", "mask", "mask_insert")):
    V, ids0 = {}, None
    for v in views:
        f = fdir / (f"{v}_generic.npz" if v == "mask_insert" else fname(v))
        z = np.load(f, allow_pickle=False)
        ids = z["ids"].astype(str)
        if ids0 is None:
            ids0 = ids
            V[v] = z["X"]
        else:
            pos = {s: j for j, s in enumerate(ids)}
            V[v] = z["X"][[pos[i] for i in ids0]]
    return V, {s: j for j, s in enumerate(ids0)}


def meta_columns(kind: str, cohort: str, c: pd.DataFrame, trap: str | None = None) -> pd.DataFrame:
    """Per image: a_any (artifact anywhere), is_free (artifact-free: may receive a pasted instance), a_in, a_out."""
    c = c.copy()
    c["image_id"] = c.image_id.astype(str)
    if kind == "trap":  # the trap pools hold artifact-free (a = 0) and trap-located artifact (a = 1) images only
        c["a_any"], c["is_free"] = c.a.astype(int), c.a.astype(int) == 0
        c["a_in"] = c.a.astype(int) if trap == "trapA" else 0
        c["a_out"] = c.a.astype(int) if trap == "trapB" else 0
        return c
    if cohort in ("thyroid", "ovary"):
        pres, free = c.marker_px >= 15, c.marker_px == 0
    elif cohort == "capsule":
        pres, free = c.contam_frac >= 0.10, c.contam_frac < 0.03
    else:  # ISIC 2019 (and the ISIC 2020 test images, which never receive pastes)
        hp = c.hair_px_native if "hair_px_native" in c else c.get("hair_px", pd.Series(0, index=c.index))
        pres, free = hp > 30, hp <= 30
        if "image_id" in c and c.image_id.str.startswith("i20_").any():
            free = free & ~c.image_id.str.startswith("i20_")
    c["a_any"], c["is_free"] = pres.astype(int), free.astype(bool)
    c["a_in"] = c.a.astype(int)
    c["a_out"] = (pres & (c.a.astype(int) == 0)).astype(int)
    return c


def build(kind: str, cohort: str, encoder: str, stage: str, smoke: bool):
    """(training-set dict {(trap, seed, fold): E}, features V, pos, meta per trap, locrand key)."""
    seeds = C.seeds_for(stage, smoke)
    if kind == "trap":
        from wtss.data.isic2019_spec import build_spec_envs
        t = R4.trap_cohort(cohort)
        c = t["c"].copy()
        V, pos = load_features(paths.CACHE / "features" / TRAP_FEAT[cohort] / BDIR[encoder])
        if smoke:  # the matched pool of a subset differs from the cached one: keep cached images only
            c = R4.smoke_subset(c[c.image_id.astype(str).isin(pos)], ("trapA", "trapB"), n=40, seed=0)
        envs = build_spec_envs(c, seeds=seeds, group_col=t["group_col"])
        missing = {str(i) for d in envs.values() for i in d.image_id} - set(pos)
        if missing:
            raise SystemExit(f"{len(missing)} trap images lack cached features (e.g. {sorted(missing)[:3]})")
        sets = {}
        for (trap, s, k, e), d in envs.items():
            sets.setdefault((trap, s, k), {})[e] = d
        meta = {}
        for trap in ("trapA", "trapB"):
            pool = pd.concat([d for (tp, _s, _k, _e), d in envs.items() if tp == trap]).drop_duplicates("image_id")
            meta[trap] = meta_columns("trap", cohort, pool, trap).set_index("image_id")
        return sets, V, pos, meta
    # natural test sets
    rn = R4.load_script("rn_r8", "scripts/run_natural.py")
    if cohort == "isic2020":
        N = _nisic()
        rn_, c19, d, _cache, _fdir19, _tau = N.load_all(smoke)
        tiers = pd.read_csv(C.ROOT / "results" / "round5" / "dedup_tiers.csv", usecols=["image_id", "drop_calibrated"])
        keep = set(tiers.loc[~tiers.drop_calibrated.astype(bool), "image_id"].astype(str))
        d = d[d.image_id.astype(str).isin(keep)].copy()
        fdir = paths.CACHE / "features" / "round5" / "isic2020_robust" / BDIR[encoder]
        if smoke:
            fdir = paths.CACHE / "features" / "round5" / "_smoke" / "isic2020_robust" / BDIR[encoder] \
                if (paths.CACHE / "features" / "round5" / "_smoke" / "isic2020_robust" / BDIR[encoder]).exists() else fdir
        V, pos = load_features(fdir)
        c19 = c19[c19.image_id.astype(str).isin(pos)].copy()
        d = d[d.image_id.astype(str).isin(pos)].copy()
        full = pd.concat([c19.assign(split="train19"), d.assign(split="test20")], ignore_index=True)
        split = lambda s: (full[full.split == "train19"], full[full.split == "test20"])
        c = full
    else:
        c, _cache, fdir, split = rn.load(cohort)
        V, pos = load_features(fdir)
    c = c[c.image_id.astype(str).isin(pos)].copy()
    c["image_id"] = c.image_id.astype(str)
    cols = ["image_id", "y", "a", "source"] if "source" in c else ["image_id", "y", "a"]
    sets = {}
    for s in seeds:
        tv, te = split(s)
        tv, te = tv[tv.image_id.astype(str).isin(pos)], te[te.image_id.astype(str).isin(pos)]
        if smoke:
            tv = tv.sample(n=min(len(tv), 600), random_state=0)
        g = tv.group.unique()
        val_g = {x for x in g if stable_int("natural_val", s, x) % 5 == 0}
        va, tr = tv[tv.group.isin(val_g)], tv[~tv.group.isin(val_g)]
        E = {"train_corr": tr, "val_clean": va, "val_groups": va, "train_all": tr, "clean": te}
        sets[("natural", s, 0)] = {k: d[cols].reset_index(drop=True).astype({"image_id": str}) for k, d in E.items()}
    meta = {"natural": meta_columns("natural", cohort, c).set_index("image_id")}
    return sets, V, pos, meta


# ------------------------------------------------------------------------------------------------ one training set
_CTX: dict = {}


def _frozen(family: str) -> str | None:
    fz = _CTX.get("frozen")
    return None if fz is None else fz["held_out_settings"].get(family)


def fit_training_set(job):
    """Fit every arm for one (trap, seed, fold); returns (choice rows, prediction frames, locrand stats)."""
    key, E = job
    trap, seed, fold = key
    V, pos, meta = _CTX["V"], _CTX["pos"], _CTX["meta"][trap]
    stage, smoke, cohort = _CTX["stage"], _CTX["smoke"], _CTX["cohort"]
    X = lambda v, d: V[v][[pos[i] for i in d.image_id.astype(str)]]
    tr, cv, vg, ta = E["train_corr"], E["val_clean"], E["val_groups"], E["train_all"]
    ytr, atr, yv = tr.y.to_numpy().astype(int), tr.a.to_numpy().astype(int), cv.y.to_numpy().astype(int)
    test_envs = [e for e in (TRAP_TEST_ENVS if trap != "natural" else ("clean",)) if e in E]
    if smoke:
        E = C.smoke_envs_from_validation(E, test_envs)
    Xm_tr, Xm_cv, Xe_tr, Xe_cv = X("mask", tr), X("mask", cv), X("erm", tr), X("erm", cv)

    # --- every setting of every family: (view, head); order = closest to masking first
    fam: dict = {}
    mask_h = CA.Head(H.fit_erm(Xm_tr, ytr, Xm_cv, yv, seed)[0])
    mb = {lam: CA.fit_weighted(Xm_tr, ytr, CA.mask_bal_weights(ytr, atr, lam), Xm_cv, yv, seed)[0] for lam in CA.LAMBDAS}
    fam["mask_bal"] = {"mask": ("mask", mask_h), **{f"lam{lam:g}": ("mask", h) for lam, h in mb.items()}}
    ids_tr = tr.image_id.astype(str).tolist()
    fam["mask_cmc"] = {"mask": ("mask", mask_h), **{m: ("mask", CA.fit_cmc(Xm_tr, ytr, atr, Xm_cv, yv, seed, m)[0])
                                                    for m in CA.CMC_MODES}}
    fam["full_cmc"] = {"mask": ("mask", mask_h), **{m: ("erm", CA.fit_cmc(Xe_tr, ytr, atr, Xe_cv, yv, seed, m)[0])
                                                    for m in CA.CMC_MODES}}
    lr_stats = {}
    if _CTX.get("paste") is not None:
        PX, pl = _CTX["paste"]
        allowed = set(ta.image_id.astype(str)) | set(ids_tr)
        version = {}
        for i in ids_tr:
            for v in pl.get(i, []):
                if v["donor"] in allowed and (v["k"], "in") in PX and i in PX[(v["k"], "in")][1]:
                    version[i] = v["k"]
                    break
        m = meta.reindex(ids_tr)
        trm = tr.assign(a_any=m.a_any.to_numpy(), recipient=(m.is_free.to_numpy().astype(bool) &
                                                              np.array([i in version for i in ids_tr])),
                        a_in=m.a_in.to_numpy(), a_out=m.a_out.to_numpy())
        paste_X = lambda k, loc, i: PX[(k, loc)][0][PX[(k, loc)][1][i]]
        for name, mode in (("locrand", "locrand"), ("locrand_loc", "loc_matched")):
            rng = np.random.default_rng(stable_int("r8_paste", cohort, trap, seed, fold, mode))
            plan = CA.paste_plan(trm, rng, mode)
            Xp = CA.paste_features(ids_tr, Xm_tr, plan, version, paste_X)
            fam[name] = {"mask": ("mask", mask_h), name: ("mask", CA.Head(H.fit_erm(Xp, ytr, Xm_cv, yv, seed)[0]))}
            lr_stats[name] = {**CA.plan_balance(trm, plan), "n_recipients": int(trm.recipient.sum()),
                              "n_in": sum(v == "in" for v in plan.values()), "n_out": sum(v == "out" for v in plan.values())}
    # descriptive families
    afr = CA.fit_afr(Xm_tr, ytr, ids_tr, Xm_cv, yv, seed)
    fam["afr"] = {"mask": ("mask", mask_h), **{k: ("mask", h) for k, h in afr.items()}}
    X0, X1 = X("mask", ta), X("mask_insert", ta)
    fam["mask_cfs"] = {"mask": ("mask", mask_h)}
    for mu in CA.CFS_MUS:
        fam["mask_cfs"][f"mu{mu:g}"] = ("mask", CA.fit_weighted(Xm_tr, ytr, CA.class_balanced_weights(ytr), Xm_cv, yv,
                                                               seed, transform=CA.cfs_transform(X0, X1, mu))[0])
    mbal1 = mb[1.0]
    fam["ens"] = {"mask": ("mask", mask_h), **{f"a{al:g}": ("mask", CA.EnsembleHead(mask_h, mbal1, al))
                                               for al in CA.ENS_ALPHAS}}
    if cohort == "isic" and trap != "natural" and _CTX.get("uncontradicted") is not None:
        fam["mask_cmc_uncontra"] = {"mask": ("mask", mask_h), **{
            m: ("mask", CA.fit_cmc(Xm_tr, ytr, atr, Xm_cv, yv, seed, m, ids_tr, _CTX["uncontradicted"])[0])
            for m in CA.CMC_MODES}}
    # fixed arms (references and descriptive)
    from wtss.methods.insertion import disease_directions, fit_difference_subspace, protect
    er = fit_difference_subspace(X0, X1, energy=0.9, max_k=64, seed=seed)
    fixed = {"erm": ("erm", CA.Head(H.fit_erm(Xe_tr, ytr, Xe_cv, yv, seed)[0])), "mask": ("mask", mask_h),
             "balanced": ("erm", CA.Head(H.fit_balanced(Xe_tr, ytr, atr, Xe_cv, yv, seed)[0])),
             "mask_balanced": ("mask", mbal1),
             "umte": ("mask", CA.Head(H.fit_erm(er(Xm_tr), ytr, er(Xm_cv), yv, seed)[0], er)),
             "umte_balanced": ("mask", CA.Head(H.fit_balanced(er(Xm_tr), ytr, atr, er(Xm_cv), yv, seed)[0], er))}
    if (atr == 0).sum() > 5 and len(np.unique(ytr[atr == 0])) == 2:
        erp = protect(er, disease_directions(Xm_tr[atr == 0], ytr[atr == 0], seed=seed))
        fixed["umte_protect"] = ("mask", CA.Head(H.fit_erm(erp(Xm_tr), ytr, erp(Xm_cv), yv, seed)[0], erp))
    g = vg
    if len(g) and min(((g.a == a) & (g.y == y)).sum() for a in (0, 1) for y in (0, 1)) > 0:
        fixed["mask_dfr"] = ("mask", CA.Head(H.fit_dfr(X("mask", g), g.y.to_numpy(), g.a.to_numpy(), Xm_cv, yv, seed)[0]))
    try:
        fixed["groupdro"] = ("mask", CA.fit_groupdro_cpu(Xm_tr, ytr, atr, Xm_cv, yv, seed)[0])
    except Exception as e:  # noqa: BLE001  (recorded, never silently)
        C.log(groupdro_failed=str(e), key=key)

    # --- selection on the selection split (val_groups)
    yg, ag = vg.y.to_numpy().astype(int), vg.a.to_numpy().astype(int)
    vp = {}

    def val_pred(view, h):
        k = (id(h), view)
        if k not in vp:
            vp[k] = h.prob(X(view, vg))
        return vp[k]

    rows, chosen = [], {}
    held_out = cohort == HELD_OUT and trap != "natural"
    for f, settings in fam.items():
        fz = _frozen(f) if held_out else None
        if held_out and stage == "confirm":
            choice = fz if fz in settings else "mask"
            rows.append({"family": f, "setting": choice, "chosen": True, "frozen": True, "frozen_setting": fz})
        else:
            val = {s: (yg, ag, val_pred(v, h)) for s, (v, h) in settings.items()}
            choice, t = C.select_setting(val, reference="mask")
            rows += [{**r, "family": f, "frozen": False} for r in t.to_dict("records")]
        chosen[f] = (choice, *settings[choice])
    # selector v2 (descriptive): the same rule across the families' chosen heads and the U-MtE arms
    members = {"mask": ("mask", mask_h)}
    for f in ("umte", "umte_protect"):
        if f in fixed:
            members[f] = fixed[f]
    for f in ("mask_bal", "mask_cmc", "full_cmc", "locrand"):
        if f in chosen:
            members[f] = chosen[f][1:]
    if not held_out:
        choice, t = C.select_setting({s: (yg, ag, val_pred(v, h)) for s, (v, h) in members.items()}, reference="mask")
        rows += [{**r, "family": "selector_v2", "frozen": False} for r in t.to_dict("records")]
    else:
        choice = "mask"
    chosen["selector_v2"] = (choice, *members[choice])
    for r in rows:
        r.update({"trap": trap, "seed": seed, "fold": fold})
    for name, st in lr_stats.items():
        rows.append({"family": name, "setting": "_plan", "trap": trap, "seed": seed, "fold": fold, **st})

    # --- predictions: validation always (selection split); test environments only in confirm / smoke
    frames = []
    envs = ["val_groups"] + (test_envs if stage == "confirm" or smoke else [])
    arms = {**{f: (s, v, h) for f, (s, v, h) in chosen.items()},
            **{f: ("fixed", v, h) for f, (v, h) in fixed.items() if f not in chosen}}
    for env in envs:
        d = E[env]
        Xv_cache = {}
        for f, (setting, view, h) in arms.items():
            if view not in Xv_cache:
                Xv_cache[view] = X(view, d)
            frames.append(pd.DataFrame({"trap": trap, "seed": seed, "fold": fold, "method": f, "setting": setting,
                                        "env": env, "image_id": d.image_id.astype(str).to_numpy(),
                                        "y": d.y.to_numpy(), "artifact_present": d.a.to_numpy(),
                                        "prob": h.prob(Xv_cache[view])}))
    return rows, frames


# ------------------------------------------------------------------------------------------------ driver
def frozen_is_committed() -> bool:
    f = C.OUT / "frozen_choice.json"
    if not f.exists():
        return False
    rel = str(f.relative_to(C.ROOT))
    tracked = subprocess.call(["git", "-C", str(C.ROOT), "ls-files", "--error-unmatch", rel],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    clean = subprocess.call(["git", "-C", str(C.ROOT), "diff", "--quiet", "HEAD", "--", rel]) == 0
    return tracked and clean


def uncontradicted_ids() -> set:
    """Images that entered the round-6 cleaned dermoscopy traps (results/round6/clean_traps/isic): no informative
    independent source contradicts their hair label. Only this output file is read (never results/round6/rating/,
    the agreement per-image tables or any audit key)."""
    p = pd.read_csv(C.ROOT / "results" / "round6" / "clean_traps" / "isic" / "predictions.csv.gz", usecols=["image_id"])
    return set(p.image_id.astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("dev", "confirm"))
    ap.add_argument("--kind", required=True, choices=("trap", "natural"))
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--encoder", default="dino518", choices=tuple(BDIR))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "dev" and a.cohort == HELD_OUT:
        raise SystemExit(f"{HELD_OUT} is the held-out cohort: no development run (prereg section 5)")
    frozen = None
    if a.stage == "confirm" and not a.smoke:
        if not frozen_is_committed():
            raise SystemExit("results/round8/frozen_choice.json must be committed before any confirmation run")
        frozen = json.loads((C.OUT / "frozen_choice.json").read_text())
    elif a.cohort == HELD_OUT and a.smoke:
        f = C.stage_dir("", True) / "frozen_choice.json"
        frozen = json.loads(f.read_text()) if f.exists() else {"held_out_settings": {}}
    out = C.stage_dir(a.stage, a.smoke) / a.kind / f"{a.cohort}_{a.encoder}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "choices.csv").exists() and (a.stage == "dev" or (out / "predictions.csv.gz").exists()):
        C.log(skip=str(out), reason="done")
        return
    sets, V, pos, meta = build(a.kind, a.cohort, a.encoder, a.stage, a.smoke)
    paste = None
    lk = LOCRAND_KEY.get(a.cohort)
    if a.encoder == "dino518" and lk is not None and (LR.feat_dir(lk, a.smoke) / "p0_in.npz").exists():
        paste = LR.load_paste_features(lk, a.smoke)
    elif a.encoder == "dino518":
        C.log(warning="no locrand features; the locrand families are not fitted", cohort=a.cohort)
    global _CTX
    _CTX = dict(V=V, pos=pos, meta=meta, stage=a.stage, smoke=a.smoke, cohort=a.cohort, frozen=frozen, paste=paste,
                uncontradicted=uncontradicted_ids() if a.cohort == "isic" and a.kind == "trap" else None)
    jobs = sorted(sets.items(), key=lambda kv: kv[0])
    if a.smoke:  # first fold of the first seed of each trap (natural: the one smoke seed)
        jobs = [j for j in jobs if j[0][2] == 0][:2]
    C.log(stage=a.stage, kind=a.kind, cohort=a.cohort, encoder=a.encoder, training_sets=len(jobs))
    res = R4.parallel_map(fit_training_set, jobs) if a.jobs > 1 else [fit_training_set(j) for j in jobs]
    rows = [r for rr, _ in res for r in rr]
    frames = [f for _, ff in res for f in ff]
    ch = pd.DataFrame(rows)
    ch.insert(0, "cohort", a.cohort)
    ch.insert(1, "encoder", a.encoder)
    ch.to_csv(out / "choices.csv", index=False, float_format="%.5f")
    p = pd.concat(frames, ignore_index=True)
    p.insert(0, "cohort", a.cohort)
    p.insert(1, "encoder", a.encoder)
    if a.stage == "dev" and not a.smoke:
        p = p[p.env == "val_groups"]
    p.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    C.log(written=str(out), choices=len(ch), prediction_rows=len(p))


if __name__ == "__main__":
    main()
