"""P5 (Pahde, Dreyer, Samek, Lapuschkin, Reveal to Revise, MICCAI 2023) step 1: score the released ISIC ResNet-18
checkpoints with their own dataset, split, artifact and metric code (third_party/r2r @ f00ce06), no training.

Their evaluation (experiments/evaluation/evaluate_by_subset_attacked.py) is followed exactly:
  split      datasets/base_dataset.py do_train_val_test_split(.1, .1): default_rng(0), val 10 %, test 10 %;
  artifact   datasets/isic_attacked.py: "Clever Hans" text, presence drawn once after np.random.seed(0) for
             attacked_classes; p_artifact = 0.1 (paper App. A.2 and their correction configs; their training generator
             says 0.8, recorded in third_party/r2r/SOURCE.md);
  variants   "ch" = as in training (artifact in p_artifact of MEL), "attacked" = artifact in every image of every
             class (p_artifact 1.0, all classes), "clean" = no artifact;
  metrics    utils/metrics.py: accuracy, macro-F1 (sklearn), AUROC (ovo) over the model's 9 outputs.
Their evaluation runs on the full split as their DataLoader yields it (batch 64, in order, no augmentation; 8 loader
workers instead of their 0, which changes nothing: the artifact is seeded per image index).

    WTSS_DATA=/workspace/data /workspace/envs/r2r/bin/python scripts/replication/p5_step1.py
Outputs: $WTSS_DATA/replication/r2r/step1/<ckpt>_<split>_<variant>.npz (logits, targets, indices) and
results/replication/r2r/step1.{csv,md}.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

R2R = "/workspace/third_party/r2r"
sys.path.insert(0, R2R)
from datasets import get_dataset  # noqa: E402
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores  # noqa: E402
from models import get_fn_model_loader  # noqa: E402

D = os.environ.get("WTSS_DATA", "/workspace/data")
DATA = [f"{D}/r2r/dataset_isic2019"]          # Train/ -> ISIC_2019_Training_Input, ISIC_2019_Training_GroundTruth.csv
CK = Path(f"{D}/r2r/checkpoints")
OUT = Path(f"{D}/replication/r2r/step1"); OUT.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).resolve().parents[2] / "results/replication/r2r"; RES.mkdir(parents=True, exist_ok=True)
P_ARTIFACT = 0.1
torch.random.manual_seed(0)   # as their script


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = get_dataset("isic_attacked")
    ds = {"ch": mk(data_paths=DATA, normalize_data=True, p_artifact=P_ARTIFACT, attacked_classes=["MEL"])}
    ds["clean"] = mk(data_paths=DATA, normalize_data=True, p_artifact=1.0, attacked_classes=[])
    ds["attacked"] = mk(data_paths=DATA, normalize_data=True, p_artifact=1.0, attacked_classes=ds["ch"].class_names)
    base = ds["ch"]
    sets = {"test": base.idxs_test, "val": base.idxs_val}
    print(f"n={len(base)} train {len(base.idxs_train)} val {len(base.idxs_val)} test {len(base.idxs_test)}; "
          f"artifact images (ch) {len(base.artifact_ids)}, of which test {np.isin(base.artifact_ids, base.idxs_test).sum()}; "
          f"MEL total {int(base.metadata.MEL.sum())}", flush=True)
    rows = []
    for ck in ("checkpoint_resnet18_isic_attacked.pth", "checkpoint_resnet18_isic_last.pth"):
        model = get_fn_model_loader("resnet18")(n_class=len(base.class_names), ckpt_path=str(CK / ck)).to(dev).eval()
        for split in ("test", "val"):
            for var in ("ch", "attacked", "clean"):
                sub = ds[var].get_subset_by_idxs(sets[split])
                outs, y = compute_model_scores(model, DataLoader(sub, batch_size=64, shuffle=False, num_workers=8), dev)
                m = compute_metrics(outs, y, base.class_names)
                np.savez_compressed(OUT / f"{ck[:-4]}_{split}_{var}.npz", logits=outs.numpy(), y=y.numpy(),
                                    idx=np.asarray(sets[split]))
                rows.append({"checkpoint": ck, "split": split, "variant": var, "n": len(y),
                             "accuracy": m["accuracy"], "f1_macro": m["f1"], "auroc_ovo": m["auc"]})
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "step1.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
