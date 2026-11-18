"""P6 (Dreyer, Pahde, Anders, Samek, Lapuschkin, "From Hope to Safety", RR-ClArC, AAAI 2024) step 1: score the released
Vanilla ISIC checkpoints (LSB attack on MEL; VGG-16, ResNet-18, EfficientNet-B0) with their own code
(third_party rrclarc @ a3e2d15), no training.

Their experiments/evaluation/evaluate_by_subset_attacked.py is followed exactly, with the configs written by their
config_files/config_generator_correcting_isic_attacked.py (Vanilla, num_epochs 0 -> the released checkpoint itself):
  split      default_rng(0), 80/10/10 (base_dataset.py), n = 25,331;
  artifact   LSB trigger (their lorem-ipsum message) in the lowest lsb_factor bits (VGG/ResNet 3, EfficientNet 2);
  variants   ch = as in training (p_artifact 0.1 of MEL, their correction configs), attacked = every image of every class
             (the paper's "biased" test), clean = none;
  metrics    utils/metrics.py accuracy and macro-F1 (the paper reports accuracy).
Batch 32 instead of their 64 (inference only; identical outputs), 8 loader workers. P6_DEVICE=cpu runs it on CPU
(used on machine 2 while the GPU is full; fp32 either way).

    /workspace/envs/r2r/bin/python scripts/replication/p6_step1.py
Outputs: $WTSS_DATA/replication/rrclarc/step1/*.npz and results/replication/rrclarc/step1.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

RC = "/workspace/third_party/rrclarc"
sys.path.insert(0, RC)
from datasets import get_dataset  # noqa: E402
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores  # noqa: E402
from models import get_fn_model_loader  # noqa: E402
from utils.artificial_artifact import get_artifact_kwargs  # noqa: E402

D = os.environ.get("WTSS_DATA", "/workspace/data")
CK = Path(f"{D}/rrclarc/checkpoints")
OUT = Path(f"{D}/replication/rrclarc/step1"); OUT.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).resolve().parents[2] / "results/replication/rrclarc"; RES.mkdir(parents=True, exist_ok=True)
torch.random.manual_seed(0)
DEV = os.environ.get("P6_DEVICE", "cuda")
torch.set_num_threads(int(os.environ.get("P6_THREADS", "20")))


def main():
    rows = []
    for arch in ("resnet18", "efficientnet_b0", "vgg16"):
        cfg = yaml.safe_load(open(f"{RC}/config_files/correcting_isic_attacked/local/{arch}_Vanilla_lsb-0.1.yaml"))
        ak = get_artifact_kwargs(cfg)
        mk = get_dataset(cfg["dataset_name"])
        ds = {"ch": mk(data_paths=cfg["data_paths"], normalize_data=True, p_artifact=cfg["p_artifact"],
                       artifact_type=cfg["artifact_type"], attacked_classes=cfg["attacked_classes"], **ak)}
        ds["clean"] = mk(data_paths=cfg["data_paths"], normalize_data=True, attacked_classes=[], **ak)
        ds["attacked"] = mk(data_paths=cfg["data_paths"], normalize_data=True, p_artifact=1.0,
                            artifact_type=cfg["artifact_type"], attacked_classes=ds["ch"].class_names, **ak)
        base = ds["ch"]
        model = get_fn_model_loader(arch)(n_class=len(base.class_names), ckpt_path=str(CK / Path(cfg["ckpt_path"]).name))
        model = model.to(DEV).eval()
        for split, idx in (("test", base.idxs_test), ("val", base.idxs_val)):
            for var in ("clean", "attacked", "ch"):
                sub = ds[var].get_subset_by_idxs(idx)
                outs, y = compute_model_scores(model, DataLoader(sub, batch_size=32, shuffle=False, num_workers=8), DEV)
                m = compute_metrics(outs, y, None)
                np.savez_compressed(OUT / f"{arch}_{split}_{var}.npz", logits=outs.numpy(), y=y.numpy(), idx=np.asarray(idx))
                rows.append({"arch": arch, "lsb_factor": cfg["lsb_factor"], "split": split, "variant": var, "n": len(y),
                             "accuracy": m["accuracy"], "f1_macro": m["f1"]})
                print(rows[-1], flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(RES / "step1.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
