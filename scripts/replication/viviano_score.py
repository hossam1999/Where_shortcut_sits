"""Per-image test scores and penultimate features of a trained P1 (Viviano et al. 2021) model.

Runs in P1's own environment (/workspace/envs/viviano). Rebuilds the seed's train/valid/test splits with their code and
config (same seed => same split, see docs/PREREGISTRATION_REPLICATION.md, amendment A1), loads the saved
best_model_<seed>.pth.tar, and writes for every image of each split:
    image_id, split, site (0/1 = NIH/PadChest or PA/AP), y, logit0, logit1
plus the 512-d global-average-pooled features (for arm a) to an .npz next to the CSV. No image or feature is committed.

    python scripts/replication/viviano_score.py --config <their yml> --seed 1234 --model results/<exp>/best_model_1234.pth.tar \
        --out $WTSS_DATA/replication/viviano/preds/<exp>_1234
The check printed at the end recomputes their test metric (roc_auc_score on one-hot targets vs the two logit columns)
for comparison with the value their training log reports at the best epoch.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import activmask.utils.configuration as configuration


def image_ids(ds):
    """Stable image identifiers for a P1 split dataset, in its own order."""
    out = []
    for site, k in zip(ds.site, ds.imageids):
        sub = ds.dataset1 if site == 0 else ds.dataset2
        csv = sub.csv
        if "Image Index" in csv.columns:            # NIH
            out.append(str(csv["Image Index"].iloc[k]))
        elif "ImageID" in csv.columns:              # PadChest
            out.append(str(csv["ImageID"].iloc[k]))
        else:                                       # RSNA (TorchXRayVision RSNA_Pneumonia_Dataset)
            out.append(str(csv["patientId"].iloc[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--mask_all", choices=["as_config", "on", "off"], default="as_config",
                    help="override the dataset mask_all flag (arm a input choice); default = their config as loaded")
    ap.add_argument("--external_roi", default=None,
                    help="dir with roi224_<split>.npz (ids, masks) from p1_arms.py --stage rois: images are multiplied by "
                         "these label-free masks after their preprocessing (amendment A11, RSNA arm a)")
    args = ap.parse_args()

    # shared GPU: cap this process so it can never push a training job out of memory
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.15")))
    cfg = configuration.load_config(args.config)
    model = torch.load(args.model, map_location="cuda", weights_only=False).cuda().eval()
    feats = {}
    model.encoder.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o.flatten(1).detach().cpu()))
    rows, F = [], {}
    for split in args.splits.split(","):
        if args.mask_all != "as_config":
            list(cfg["dataset"][split].values())[0]["mask_all"] = args.mask_all == "on"
        ds = configuration.setup_dataset(cfg, args.seed, split=split)(configuration.setup_transform(cfg, split))
        ids = image_ids(ds)
        ext = None
        if args.external_roi:
            z = np.load(f"{args.external_roi}/roi224_{split}.npz")
            pos = {k: j for j, k in enumerate(z["ids"].astype(str))}
            ext = torch.from_numpy(z["masks"][[pos[i] for i in ids]].astype(np.float32))[:, None]
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=16)
        logits, fs, ys = [], [], []
        k0 = 0
        with torch.no_grad():
            for X, seg, y in dl:
                if ext is not None:
                    X = X * torch.nn.functional.interpolate(ext[k0:k0 + len(X)], size=X.shape[-2:], mode="nearest")
                k0 += len(X)
                X = X.cuda()
                seg = torch.abs(seg - 1).type(torch.BoolTensor).cuda()
                logits.append(model.forward(X, seg)["y_pred"].cpu())
                fs.append(feats["f"])
                ys.append(y)
        L, Y = torch.cat(logits).numpy(), torch.cat(ys).numpy()
        F[split] = torch.cat(fs).numpy().astype(np.float32)
        rows.append(pd.DataFrame({"image_id": ids, "split": split, "site": ds.site, "y": Y,
                                  "logit0": L[:, 0], "logit1": L[:, 1]}))
        if split == "test":
            oh = OneHotEncoder(sparse_output=False, categories="auto").fit_transform(Y.reshape(-1, 1))
            print(f"test AUROC (their definition, one-hot macro): {roc_auc_score(oh, L):.4f}; "
                  f"logit-difference AUROC: {roc_auc_score(Y, L[:, 1] - L[:, 0]):.4f}; n={len(Y)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.concat(rows).to_csv(args.out + ".csv.gz", index=False)
    np.savez_compressed(args.out + "_features.npz", **F)


if __name__ == "__main__":
    main()
