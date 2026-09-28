"""Round 8, D5: controlled sweeps at r = 0 and r = 1, cross-fitted over every cohort image
(docs/PREREGISTRATION_ROUND8.md, section 7, D5).

  python scripts/round8/sweeps.py --sweep thyroid [--smoke]

The earlier sweeps scored one fixed test split (about 216 images per overlap and seed), too few for a non-inferiority
margin. Here every image of the sweep cohort is a test image once: 5-fold cross-fitting (StratifiedGroupKFold by label
and pHash group) with new artifact-presence seeds (confirmation seeds). Features are the cached renders of every cohort
image (clean, and with the synthetic artifact at overlap 0 and 1; views erm and mask), so no image is encoded again.
Protocol otherwise as the sweeps of Stage 2: training presence 0.9 / 0.1, correlated 0.9 / 0.1 and reversed 0.1 / 0.9
test environments, clean test; C chosen on the clean inner-validation images; the selection split is the inner
validation images with the 0.5 / 0.5 ("test_uncorr") artifact assignment. Candidates: mask_bal(λ), mask_cmc, full_cmc
(locrand needs real instances and is not defined for synthetic artifacts); references erm, mask, mask_balanced.
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
CA = _r8("candidates")
R4 = C.load_round4_common()

from wtss import heads as H  # noqa: E402
from wtss import paths  # noqa: E402
from wtss.utils import stable_int  # noqa: E402

# sweep -> (cohort name for run_synthetic.cohort_and_placements, feature cohort dir, artifact); DINOv2-B/14 @ 518
SWEEPS = {"thyroid": ("thyroid", "thyroid_synth", "caliper"), "capsule": ("capsule", "capsule_synth", "debris"),
          "ovary": ("ovary", "ovary_synth", "caliper"), "isic2018": ("isic2018", "isic2018_pilot", "ruler_fixed")}
# The NIH chest-tube sweep is not run: its cohort table (labels) is not on this machine, only its cached features.
BDIR = "dinov2_b14_518"
OVERLAPS = (0.0, 1.0)


def load(sweep: str):
    from wtss.features import assemble_env  # noqa: F401
    rs = R4.load_script("run_synth_r8", "scripts/run_synthetic.py")
    name, fcoh, art = SWEEPS[sweep]
    cohort = rs.cohort_and_placements(name, 518)[0]
    df = cohort.df.copy()
    df["image_id"] = df.image_id.astype(str)
    fd = paths.CACHE / "features" / fcoh / BDIR / art
    V = {}
    for view in ("erm", "mask"):
        for tag in ("clean", "ov000", "ov100"):
            z = np.load(fd / f"{view}_{tag}.npz", allow_pickle=False)
            pos = {s: j for j, s in enumerate(z["ids"].astype(str))}
            V[(view, tag)] = z["X"][[pos[i] for i in df.image_id]]
    if "group" not in df:
        df["group"] = df.image_id
    return df.reset_index(drop=True), V


_CTX: dict = {}


def fit_job(job):
    ov, seed, fold, te_idx, tr_idx, va_idx = job
    from wtss.features import assemble_env
    from wtss.synthetic import presence_vector
    df, V = _CTX["df"], _CTX["V"]
    ids, y = df.image_id.to_numpy(), df.y.to_numpy().astype(int)
    pres = {e: presence_vector(ids, y, seed, e).astype(int) for e in ("train_corr", "test_corr", "test_rev", "test_uncorr")}
    tag = "ov000" if ov == 0.0 else "ov100"
    Xenv = lambda view, env: assemble_env(V[(view, "clean")], V[(view, tag)], pres[env].astype(bool))
    Xtr = {v: Xenv(v, "train_corr")[tr_idx] for v in ("erm", "mask")}
    Xcv = {v: V[(v, "clean")][va_idx] for v in ("erm", "mask")}          # C selection: clean validation images
    Xsel = {v: Xenv(v, "test_uncorr")[va_idx] for v in ("erm", "mask")}  # selection split: 50/50 assignment
    ytr, atr, yv = y[tr_idx], pres["train_corr"][tr_idx], y[va_idx]
    ysel, asel = y[va_idx], pres["test_uncorr"][va_idx]
    mask_h = CA.Head(H.fit_erm(Xtr["mask"], ytr, Xcv["mask"], yv, seed)[0])
    mb = {lam: CA.fit_weighted(Xtr["mask"], ytr, CA.mask_bal_weights(ytr, atr, lam), Xcv["mask"], yv, seed)[0]
          for lam in CA.LAMBDAS}
    fam = {"mask_bal": {"mask": ("mask", mask_h), **{f"lam{k:g}": ("mask", h) for k, h in mb.items()}},
           "mask_cmc": {"mask": ("mask", mask_h), **{m: ("mask", CA.fit_cmc(Xtr["mask"], ytr, atr, Xcv["mask"], yv, seed, m)[0])
                                                     for m in CA.CMC_MODES}},
           "full_cmc": {"mask": ("mask", mask_h), **{m: ("erm", CA.fit_cmc(Xtr["erm"], ytr, atr, Xcv["erm"], yv, seed, m)[0])
                                                     for m in CA.CMC_MODES}}}
    rows, arms = [], {"erm": ("fixed", "erm", CA.Head(H.fit_erm(Xtr["erm"], ytr, Xcv["erm"], yv, seed)[0])),
                      "mask": ("fixed", "mask", mask_h), "mask_balanced": ("fixed", "mask", mb[1.0])}
    for f, settings in fam.items():
        val = {s: (ysel, asel, h.prob(Xsel[v])) for s, (v, h) in settings.items()}
        choice, t = C.select_setting(val, reference="mask")
        rows += [{**r, "family": f, "overlap": ov, "seed": seed, "fold": fold} for r in t.to_dict("records")]
        arms[f] = (choice, *settings[choice])
    frames = []
    test_sets = {"test_corr": pres["test_corr"], "test_rev": pres["test_rev"], "clean": np.zeros(len(y), int)}
    idx = va_idx if _CTX["smoke"] else te_idx  # smoke never scores a test-fold image
    for env, pv in test_sets.items():
        feats = {v: (V[(v, "clean")] if env == "clean" else Xenv(v, env))[idx] for v in ("erm", "mask")}
        for f, (setting, view, h) in arms.items():
            frames.append(pd.DataFrame({"overlap": ov, "seed": seed, "fold": fold, "method": f, "setting": setting,
                                        "env": env, "image_id": ids[idx], "y": y[idx], "artifact_present": pv[idx],
                                        "prob": h.prob(feats[view])}))
    return rows, frames


def folds(df: pd.DataFrame, seed: int, smoke: bool):
    from sklearn.model_selection import StratifiedGroupKFold
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    out = []
    for k, (rest, te) in enumerate(sg.split(df, df.y, df.group)):
        g = df.group.to_numpy()[rest]
        vg = {x for x in np.unique(g) if stable_int("r8_sweep_val", seed, k, x) % 5 == 0}
        va = rest[np.isin(g, list(vg))]
        tr = rest[~np.isin(g, list(vg))]
        out.append((k, te, tr, va))
        if smoke:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, choices=tuple(SWEEPS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    out = C.stage_dir("confirm", a.smoke) / "sweeps" / a.sweep
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predictions.csv.gz").exists():
        C.log(skip=str(out))
        return
    df, V = load(a.sweep)
    if a.smoke:
        keep = df.groupby("y", group_keys=False).apply(lambda g: g.head(150)).index
        df = df.loc[keep].reset_index(drop=True)
        V = {k: v[keep.to_numpy()] for k, v in V.items()}
    global _CTX
    _CTX = dict(df=df, V=V, smoke=a.smoke)
    seeds = C.SMOKE_SEEDS if a.smoke else C.CONF_SEEDS
    jobs = [(ov, s, k, te, tr, va) for ov in OVERLAPS for s in seeds for (k, te, tr, va) in folds(df, s, a.smoke)]
    C.log(sweep=a.sweep, images=len(df), jobs=len(jobs))
    res = R4.parallel_map(fit_job, jobs)
    ch = pd.DataFrame([r for rr, _ in res for r in rr])
    ch.insert(0, "sweep", a.sweep)
    ch.to_csv(out / "choices.csv", index=False, float_format="%.5f")
    p = pd.concat([f for _, ff in res for f in ff], ignore_index=True)
    p.insert(0, "sweep", a.sweep)
    p.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    C.log(written=str(out), rows=len(p))


if __name__ == "__main__":
    main()
