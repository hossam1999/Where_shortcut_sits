"""Render P2's images exactly as their network sees them, before normalisation (arms b, c; section 3): their dataset
code at 224 px with the microscope inserted by their rule (train "as trained": 20% of MEL; val/test clean: none; biased:
every image), through the same subsets and indices as p2_score.py. CPU only (pahde env).

    python scripts/replication/p2_render.py --config <their Vanilla yaml>
Writes $WTSS_DATA/replication/pahde/render224/<set>.npy (N x 224 x 224 x 3 uint8) and <set>.csv (index, image, label,
artifact) for set in train_as_trained, val_clean, val_biased, test_clean, test_biased.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

P = "/workspace/third_party/pahde"
sys.path.insert(0, P)
os.chdir(P)
from datasets import load_dataset  # noqa: E402

D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = Path(f"{D}/replication/pahde/render224")


def collate(b):
    return np.stack([np.asarray(x[0], np.uint8) for x in b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    base = load_dataset(cfg)
    clean = load_dataset({**copy.deepcopy(cfg), "p_artifact": 0.0})
    attacked = load_dataset({**copy.deepcopy(cfg), "attacked_classes": base.classes, "p_artifact": 1.0})
    OUT.mkdir(parents=True, exist_ok=True)
    sets = [("train_as_trained", base.idxs_train, base), ("val_clean", base.idxs_val, clean), ("val_biased", base.idxs_val, attacked),
            ("test_clean", base.idxs_test, clean), ("test_biased", base.idxs_test, attacked)]
    for name, idxs, ds in sets:
        if (OUT / f"{name}.csv").exists():
            continue
        sub = ds.get_subset_by_idxs(idxs)
        sub.transform, sub.do_augmentation = None, False
        mm = np.lib.format.open_memmap(OUT / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(len(idxs), 224, 224, 3))
        dl = torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=16, collate_fn=collate)
        k = 0
        for x in dl:
            mm[k:k + len(x)] = x; k += len(x)
        mm.flush(); del mm
        ys = [int(sub.get_target(j)) for j in range(len(sub))] if hasattr(sub, "get_target") else None
        pd.DataFrame({"index": np.asarray(idxs), "image": base.metadata.image.iloc[np.asarray(idxs)].to_numpy(),
                      "label": ys, "artifact": np.asarray(ds.artifact_labels)[np.asarray(idxs)].astype(int)}).to_csv(OUT / f"{name}.csv", index=False)
        print(name, k, flush=True)


if __name__ == "__main__":
    main()
