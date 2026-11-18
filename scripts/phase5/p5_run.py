"""Phase 5 (thyroid TN3K calipers; plan "Phase 5, minimal scope"): train one network on one trap / fold and write its
predictions. Library and rules: src/wtss/experiments/phase5.py. Environments: the thesis spec traps (seed 42, folds 0-2;
build_spec_envs on scripts/run_thyroid_traps.cohort(), groups = pHash leakage groups), images from the 518-px thyroid cache.

    python scripts/phase5/p5_run.py --net erm_r50d --trap trapA --fold 1
    python scripts/phase5/p5_run.py --net rrclarc --trap trapA --fold 1 --lam 5e9 --layer identity_2
    python scripts/phase5/p5_run.py --net isnet   --trap trapA --fold 1
    python scripts/phase5/p5_run.py --net armd    --trap trapA --fold 1
    add --max_batches N for the timing test (no validation, no predictions)

Outputs (never committed): $WTSS_DATA/phase5/<trap>_f<fold>/<net>[_lam<lam>_<layer>]/ {best.pt|last.pt, log.csv, preds.csv.gz,
features.npz (armd)}. Test predictions are written once here; every AUROC is computed later by the registered analysis.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wtss.data.isic2019_spec import build_spec_envs  # noqa: E402
from wtss.experiments import phase5 as P5  # noqa: E402
from wtss.experiments.real_traps import RealCache, make_renderers  # noqa: E402

D = Path(os.environ.get("WTSS_DATA", "/workspace/data"))
DEFAULTS = {"erm_r50d": dict(lr=1e-4, bs=48), "rrclarc": dict(lr=1e-5, bs=48), "isnet": dict(lr=1e-3, bs=16),
            "armd": dict(lr=1e-4, bs=64)}
ENVS = ("train_corr", "val_clean", "test_corr", "test_rev", "clean")


def thyroid():
    spec = importlib.util.spec_from_file_location("rt", ROOT / "scripts" / "run_thyroid_traps.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    c = rt.cohort()
    envs = build_spec_envs(c, seeds=(42,), group_col="group")
    cache = RealCache(rt.T / "cache_518", roi_file="roi.npy", art_file="marker.npy")
    return envs, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(DEFAULTS), required=True)
    ap.add_argument("--trap", choices=["trapA", "trapB"], required=True)
    ap.add_argument("--fold", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--layer", default=None, choices=[None, "identity_1", "identity_2", "last_conv"])
    ap.add_argument("--isnet_pretrained", action="store_true", help="ImageNet DenseNet-121 (decision in PREREGISTRATION_PHASE5)")
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.fold); np.random.seed(a.fold)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    envs, cache = thyroid()
    E = {e: envs[(a.trap, 42, a.fold, e)] for e in ENVS}
    render = make_renderers(cache, 518, [])
    roi_fn = lambda i: cache.get(i)[1]   # noqa: E731
    view = "mask" if a.net == "armd" else "erm"
    norm = "unit" if (a.net == "isnet" and not a.isnet_pretrained) else "imagenet"
    mk = lambda df, train: P5.P5Images(df, render, view, 224, train, norm, roi_fn if a.net == "isnet" else None, seed=a.fold)  # noqa: E731
    d = DEFAULTS[a.net]
    cfg = P5.TrainCfg(epochs=a.epochs, patience=a.patience, bs=d["bs"], workers=a.workers, seed=a.fold, max_batches=a.max_batches, lr=d["lr"])
    name = a.net + (f"_lam{a.lam:g}_{a.layer}" if a.net == "rrclarc" else "") + (f"_{a.tag}" if a.tag else "")
    out = D / "phase5" / f"{a.trap}_f{a.fold}" / name
    tr, va = mk(E["train_corr"], True), mk(E["val_clean"], False)
    rr, fixed = None, None
    if a.net == "erm_r50d":
        model = P5.make_resnet50d(True)
    elif a.net == "armd":
        model = P5.make_effb0(True)
    elif a.net == "isnet":
        model = P5.make_isnet(a.isnet_pretrained)
    else:   # rrclarc: weights only from this fold's ERM base, fresh optimiser, their 10 epochs
        base = D / "phase5" / f"{a.trap}_f{a.fold}" / "erm_r50d" / "best.pt"
        model = P5.make_resnet50d(False); model.load_state_dict(torch.load(base, map_location="cpu")); model.to(dev)
        acts = P5.collect_cavs_max(model, a.layer, P5.loader(mk(E["train_corr"], False), 64, False, a.workers), dev)
        cav = P5.compute_rrclarc_cav(acts, E["train_corr"].a.to_numpy(), "signal")
        out.mkdir(parents=True, exist_ok=True); torch.save(cav, out / "cav.pt")
        rr, fixed = P5.RRClarcStep(model, a.layer, cav, a.lam), 10
    model.to(dev)
    res = P5.fit(a.net, model, tr, va, cfg, out, dev, rr=rr, fixed_epochs=fixed)
    print(json.dumps({"net": name, "trap": a.trap, "fold": a.fold, **res}), flush=True)
    if a.max_batches:
        return
    model.load_state_dict(torch.load(out / ("last.pt" if fixed else "best.pt"), map_location="cpu")); model.to(dev)
    frames, feats = [], {}
    for e in ENVS:
        ds = mk(E[e], False); dl = P5.loader(ds, 64, False, a.workers)
        p = P5.predict(a.net, model, dl, dev)
        frames.append(E[e][["image_id", "y", "a"]].assign(env=e, prob=p))
        if a.net == "armd":
            with torch.no_grad():
                model.eval()
                feats[e] = np.concatenate([P5.effb0_features(model, x.to(dev)).cpu().numpy() for x, _, _, _ in dl])
    pd.concat(frames).assign(net=name, trap=a.trap, fold=a.fold).to_csv(out / "preds.csv.gz", index=False)
    if feats:
        np.savez(out / "features.npz", **{f"X_{e}": v for e, v in feats.items()},
                 **{f"y_{e}": E[e].y.to_numpy() for e in ENVS}, **{f"a_{e}": E[e].a.to_numpy() for e in ENVS},
                 **{f"id_{e}": E[e].image_id.to_numpy().astype("U") for e in ENVS})
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
