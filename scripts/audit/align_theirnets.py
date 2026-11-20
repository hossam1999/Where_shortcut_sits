"""Alignment audit, arm-a feature files of their networks (author, 2026-11-20; after A30) — CPU.
  --part p1loc (P1 env): location test, inside / outside, arm a = their Standard network's features of circle-masked
         images (p1_location.py score), every environment: recompute 200 rows with the same dataset class and network.
  --part p2 (P2 env): arm a = our Vanilla's features (p2_score --features), arm a' = the selected A18 RR-ClArC's features
         (p2_aprime_features), every set: recompute 200 rows through their dataset subsets.
Appends to results/audit/align_theirnets.csv (cosine >= 0.99 with the cached row; row ids vs the score file's ids).
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
D = "/workspace/data"
N = 200
torch.set_num_threads(32)
rng = np.random.default_rng(20261120)
OUT = ROOT / "results/audit/align_theirnets.csv"


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def save(row):
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])]).to_csv(OUT, index=False); print(row, flush=True)


def p1loc():
    sys.path[:0] = [str(ROOT / "scripts" / "replication")]
    import p1_location as P
    cls = P.make_dataset_class()
    for pos in ("inside", "outside"):
        L = P.L / pos
        model = torch.load(L / f"results/xray-location_{pos}/best_model_{P.SEED}.pth.tar", map_location="cpu", weights_only=False).eval()
        f = {}
        model.encoder.avgpool.register_forward_hook(lambda m, i, o: f.__setitem__("z", o.flatten(1)))
        F = np.load(L / "features_masked.npz"); sc = pd.read_csv(L / "scores.csv.gz")
        for env in ("train", "valid", "corr", "rev", "clean"):
            split = env if env in ("train", "valid") else "test"
            ds = cls(position=pos, env=env, mode=split, seed=P.SEED, ratio=0.5, mask_all=True, verbose=False, **P.PATHS)
            ids_ok = bool((np.asarray(ds.keys) == sc[(sc.method == "masked") & (sc.test_set == env)].image_id.to_numpy()).all())
            pick = rng.choice(len(ds), min(N, len(ds)), replace=False); bad = 0
            with torch.no_grad():
                for j in pick:
                    X, seg, _ = ds[int(j)]
                    model.forward(X[None], torch.abs(seg[None] - 1).type(torch.BoolTensor))
                    bad += cos(f["z"][0].numpy(), F[env][j]) < 0.99
            save({"setting": f"P1 location {pos}", "split": env, "arm": "a (their Standard net, circle-masked)", "rows": len(ds),
                  "checked": len(pick), "ids_order_ok": ids_ok, "feature_mismatch": int(bad)})


def p2():
    sys.path.insert(0, "/workspace/third_party/pahde"); os.chdir("/workspace/third_party/pahde")
    import yaml
    from datasets import load_dataset
    from models import get_fn_model_loader
    R = Path(f"{D}/replication/pahde")
    sel = pd.read_csv(ROOT / "results/replication/pahde/rrclarc_selection_a18.csv"); ch = sel[sel.chosen].model.iloc[0]
    for arm, cfgp, ckpt, feats, scores in (
            ("a (our Vanilla)", R / "runs/vanilla_resnet50d_seed0/config.yaml", R / "runs/vanilla_resnet50d_seed0/checkpoints/config/checkpoint_resnet50d_last.pth",
             R / "scores/vanilla_seed0_features.npz", R / "scores/vanilla_seed0.csv.gz"),
            ("a' (A18 RR-ClArC)", R / f"mitigation_a18/{ch}.yaml", R / f"checkpoints_corrected_a18/{ch}/last.ckpt",
             R / "scores_a18/aprime_features_unmasked.npz", R / "scores_a18/aprime_unmasked_scores.csv.gz")):
        cfg = yaml.safe_load(open(cfgp))
        base = load_dataset(cfg); clean = load_dataset({**copy.deepcopy(cfg), "p_artifact": 0.0})
        attacked = load_dataset({**copy.deepcopy(cfg), "attacked_classes": base.classes, "p_artifact": 1.0})
        model = get_fn_model_loader(model_name=cfg["model_name"])(n_class=len(base.classes), ckpt_path=str(ckpt), device="cpu").eval()
        f = {}
        model.fc.register_forward_hook(lambda m, i, o: f.__setitem__("z", i[0]))
        F = np.load(feats); s = pd.read_csv(scores, usecols=["index", "split", "test_set"])
        for k, idxs, ds, sp, ts in (("train_as_trained", base.idxs_train, base, "train", "as_trained"), ("val_clean", base.idxs_val, clean, "val", "clean"),
                                    ("val_biased", base.idxs_val, attacked, "val", "biased"), ("test_clean", base.idxs_test, clean, "test", "clean"),
                                    ("test_biased", base.idxs_test, attacked, "test", "biased")):
            sub = ds.get_subset_by_idxs(idxs); sub.do_augmentation = False
            ids_ok = bool((np.asarray(idxs) == s[(s.split == sp) & (s.test_set == ts)]["index"].to_numpy()).all())
            pick = rng.choice(len(sub), min(N, len(sub)), replace=False); bad = 0
            with torch.no_grad():
                for j in pick:
                    model(sub[int(j)][0][None]); bad += cos(f["z"][0].numpy(), F[k][j]) < 0.99
            save({"setting": "P2", "split": k, "arm": arm, "rows": len(sub), "checked": len(pick), "ids_order_ok": ids_ok, "feature_mismatch": int(bad)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["p1loc", "p2"], required=True)
    {"p1loc": p1loc, "p2": p2}[ap.parse_args().part]()
