"""Round 8, D6: location randomisation during fine-tuning of the thyroid ConvNeXt-T (docs/PREREGISTRATION_ROUND8.md,
section 7, D6).

  python scripts/round8/ft.py --part natural [--smoke]    # official thyroid split, confirmation seeds
  python scripts/round8/ft.py --part traps   [--smoke]    # thyroid traps, env seeds 8101 and 8202 x 5 folds

Recipe fixed by round 6 on validation only (results/round6/ft_recipe.json: convnext_tiny.fb_in22k_ft_in1k, lr 1e-4,
8 epochs). Arms: erm, mask (references, refitted with the new seeds) and locrand_ft: the masking arm trained on images in
which real caliper instances were pasted into artifact-free training images exactly as the frozen-feature locrand
(paste plan of candidates.paste_plan, versions from scripts/round8/locrand.py with training-set donors only). Test and
validation images are never pasted. Selection (fallback to masking) by the registered rule on the validation split.
"""
from __future__ import annotations

import os
import argparse
import importlib.util
import json
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

from wtss.utils import stable_int  # noqa: E402

ARMS = ("erm", "mask", "locrand_ft")


def recipe() -> dict:
    return json.loads((C.ROOT / "results" / "round6" / "ft_recipe.json").read_text())["choice"]


def paste_render(base: dict, cache, kind: str, pl: dict, plan: dict, version: dict):
    """Render dict whose 'mask' view returns the pasted-then-masked image for planned training images, else the plain
    masked image (every validation and test image)."""
    rends = {}

    def mask(i):
        loc = plan.get(str(i))
        if loc is None:
            return base["mask"](i)
        k = version[str(i)]
        if (k, loc) not in rends:
            rends[(k, loc)] = LR.make_paste_renderer(cache, kind, pl, k, loc, 518, masked=True)
        return rends[(k, loc)](i)

    r = dict(base)
    r["mask"] = mask
    return r


def locrand_plan(tr: pd.DataFrame, info: pd.DataFrame, pl: dict, allowed: set, key: tuple):
    ids = tr.image_id.astype(str).tolist()
    version = {}
    for i in ids:
        for v in pl.get(i, []):
            if v["donor"] in allowed:
                version[i] = v["k"]
                break
    m = info.reindex(ids)
    trm = tr.assign(a_any=m.present.astype(int).to_numpy(),
                    recipient=m.recipient.fillna(False).astype(bool).to_numpy() & np.array([i in version for i in ids]))
    plan = CA.paste_plan(trm, np.random.default_rng(stable_int("r8_ft_paste", *key)), "locrand")
    return plan, version, CA.plan_balance(trm, plan)


def run(part: str, smoke: bool):
    import torch
    from wtss.evaluation import evaluate  # noqa: F401
    from wtss.experiments.finetune import train_eval
    from wtss.experiments.real_traps import make_renderers
    rc = recipe()
    epochs = 1 if smoke else int(rc["epochs"])
    info, cache, kind = LR.cohort_info("thyroid")
    info = info.set_index("image_id")
    pl = LR.placements("thyroid", smoke)
    base = make_renderers(cache, 518, [])
    out = C.stage_dir("confirm", smoke) / "ft" / part
    out.mkdir(parents=True, exist_ok=True)
    pf = out / "predictions.csv.gz"
    done = pd.read_csv(pf) if pf.exists() else None
    frames = [] if done is None else [done]
    stats = []
    if part == "natural":
        ft6 = R4.load_script("ft_r6", "scripts/round6/ft_thyroid.py")
        c, _cache, _, _ = ft6._natural()
        units = []
        for s in (C.SMOKE_SEEDS if smoke else C.CONF_SEEDS):
            tr, va, te = ft6.parts(c, s, smoke)
            units.append((("natural", s, 0), tr, va, {"clean": va if smoke else te, "val_groups": va}))
    else:
        from wtss.data.isic2019_spec import build_spec_envs
        rt = R4.load_script("rt", "scripts/run_thyroid_traps.py")
        c = rt.cohort()
        if smoke:
            c = R4.smoke_subset(c, ("trapA", "trapB"), n=40, seed=0)
        seeds = C.SMOKE_SEEDS if smoke else C.FT_TRAP_SEEDS
        envs = build_spec_envs(c, seeds=seeds, group_col="group")
        units = []
        for trap in ("trapA", "trapB"):
            for s in seeds:
                for k in ((0,) if smoke else range(5)):
                    E = {e: envs[(trap, s, k, e)] for e in ("train_corr", "val_clean", "val_groups", "train_all",
                                                           "test_corr", "test_rev", "clean")}
                    tests = {e: E["val_groups"] if smoke else E[e] for e in ("test_corr", "test_rev", "clean")}
                    tests["val_groups"] = E["val_groups"]
                    units.append(((trap, s, k), E["train_corr"], E["val_clean"], tests, E["train_all"]))
    for u in units:
        key, tr, va, tests = u[0], u[1], u[2], u[3]
        ta = u[4] if len(u) > 4 else tr
        cid = key[1] * 10 + key[2]
        allowed = set(ta.image_id.astype(str)) | set(tr.image_id.astype(str))
        plan, version, st = locrand_plan(tr, info, pl, allowed, key)
        stats.append({"trap": key[0], "seed": key[1], "fold": key[2], **st, "n_in": sum(v == "in" for v in plan.values())})
        # all evaluation sets in one pass: rows keep their env tag
        ev = pd.concat([d.assign(_env=e) for e, d in tests.items()], ignore_index=True)
        # train_eval also predicts test_corr / test_rev: give it two validation rows (discarded)
        E = {"train_corr": tr, "clean_val": va, "clean_test": ev, "test_corr": va.head(2), "test_rev": va.head(2)}
        for arm in ARMS:
            if done is not None and ((done.trap == key[0]) & (done.seed == cid) & (done.method == arm)).any():
                continue
            render = paste_render(base, cache, kind, pl, plan, version) if arm == "locrand_ft" else base
            res = train_eval("mask" if arm == "locrand_ft" else arm, E, render, arch=rc["arch"], epochs=epochs,
                             lr=float(rc["lr"]), seed=cid, device=torch.device("cuda"), workers=6)
            clf, _thr, d = res["clean_test"]
            p = clf.predict_proba(None)[:, 1]
            frames.append(pd.DataFrame({"trap": key[0], "seed": cid, "fold": key[2], "method": arm, "env": d._env.to_numpy(),
                                        "image_id": d.image_id.astype(str).to_numpy(), "y": d.y.to_numpy(),
                                        "artifact_present": d.a.to_numpy(), "prob": p}))
            pd.concat(frames, ignore_index=True).to_csv(pf, index=False, compression="gzip")
            C.log(ft=part, key=key, arm=arm)
    pd.DataFrame(stats).to_csv(out / "paste_plans.csv", index=False)
    # selection (fallback to masking) per training set, on the validation split
    p = pd.concat(frames, ignore_index=True)
    rows = []
    for (trap, seed), q in p[p.env == "val_groups"].groupby(["trap", "seed"]):
        val = {}
        for m, name in (("mask", "mask"), ("locrand_ft", "locrand_ft")):
            r = q[q.method == m]
            if len(r):
                val[name] = (r.y.to_numpy(), r.artifact_present.to_numpy(), r.prob.to_numpy())
        if "mask" in val:
            choice, t = C.select_setting(val, reference="mask")
            rows += [{**x, "trap": trap, "seed": seed} for x in t.to_dict("records")]
    pd.DataFrame(rows).to_csv(out / "choices.csv", index=False, float_format="%.5f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("natural", "traps"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.smoke:
        HR = _r8("heads_run")
        if not HR.frozen_is_committed():
            raise SystemExit("results/round8/frozen_choice.json must be committed before any confirmation run")
    run(a.part, a.smoke)


if __name__ == "__main__":
    main()
