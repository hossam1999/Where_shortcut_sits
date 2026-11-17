"""Per-image scores of a P2 (Pahde et al. 2025) model on their validation and test splits, clean and biased.

Runs in P2's environment. Datasets are built exactly as their trainer (`model_training/start_training.py`): clean =
`p_artifact = 0`; biased ("attacked") = artifact inserted in every class with `p_artifact = 1`; subsets by their
`idxs_val` / `idxs_test` (rng seed 0). The model is loaded with their `get_fn_model_loader` (as their
`experiments/evaluation/evaluate_by_subset_attacked.py`, which scores corrected models from `last.ckpt`).

    python scripts/replication/p2_score.py --config <their yaml> --ckpt <checkpoint> --out <csv.gz> [--features]
Writes image index, split, test set (clean / biased), label, all class logits, prediction; optional penultimate features
(.npz) for arm a.
"""
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "/workspace/third_party/pahde")
from datasets import load_dataset  # noqa: E402
from models import get_fn_model_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.2")))
    cfg = yaml.safe_load(open(a.config))
    base = load_dataset(cfg)
    clean = load_dataset({**copy.deepcopy(cfg), "p_artifact": 0.0})
    attacked = load_dataset({**copy.deepcopy(cfg), "attacked_classes": base.classes, "p_artifact": 1.0})
    model = get_fn_model_loader(model_name=cfg["model_name"])(n_class=len(base.classes), ckpt_path=a.ckpt, device="cuda").cuda().eval()
    feats = {}
    if a.features:
        model.fc.register_forward_hook(lambda m, i, o: feats.__setitem__("f", i[0].detach().cpu()))
    rows, F = [], {}
    sets = [("val", base.idxs_val, "clean", clean), ("val", base.idxs_val, "biased", attacked),
            ("test", base.idxs_test, "clean", clean), ("test", base.idxs_test, "biased", attacked)]
    if a.features:   # arm a (mask_cmc) is fitted on the training set as their trainer sees it (artifact in 20% of MEL)
        sets = [("train", base.idxs_train, "as_trained", base)] + sets
    for split, idxs, name, ds in sets:
        if True:
            sub = ds.get_subset_by_idxs(idxs)
            sub.do_augmentation = False
            dl = torch.utils.data.DataLoader(sub, batch_size=32, shuffle=False, num_workers=12)
            lg, ys, fs = [], [], []
            with torch.no_grad():
                for x, y in dl:
                    lg.append(model(x.cuda()).float().cpu()); ys.append(y)
                    if a.features:
                        fs.append(feats["f"])
            L = torch.cat(lg).numpy(); Y = torch.cat(ys).numpy()
            d = pd.DataFrame({"index": np.asarray(idxs), "image": base.metadata.image.iloc[np.asarray(idxs)].to_numpy(),
                              "split": split, "test_set": name, "label": Y, "pred": L.argmax(1),
                              "artifact": np.asarray(ds.artifact_labels)[np.asarray(idxs)].astype(int)})
            for k in range(L.shape[1]):
                d[f"logit{k}"] = L[:, k]
            rows.append(d)
            if a.features:
                F[f"{split}_{name}"] = torch.cat(fs).numpy()
            print(split, name, "accuracy", round(float((d.pred == d.label).mean()), 4), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.concat(rows).to_csv(a.out, index=False)
    if a.features:
        np.savez_compressed(a.out.replace(".csv.gz", "_features.npz"), **F)


if __name__ == "__main__":
    main()
