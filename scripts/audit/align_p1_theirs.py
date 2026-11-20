"""Alignment audit (author, 2026-11-20; after A30), P1 their pipeline and arm a's features — CPU, P1's environment.

For cardiomegaly and RSNA (seed 1234), per split:
  1. row order: the ids of P1's split dataset (their code) == the Standard score file's rows == arm a's score file rows
     (arm a takes features from the Masked / ActDiff run but labels from the Standard file);
  2. image <-> id (200 random rows): the image their dataset returns for row j is the file named by row j's id
     (independent load, same size; Pearson r >= 0.97 and higher than for every other sampled file);
  3. label / site <-> id: from the source tables (NIH Data_Entry "Cardiomegaly", PadChest labels "cardiomegaly";
     RSNA stage_2_train_labels Target; site = NIH/PadChest or the RSNA view from TorchXRayVision's metadata);
  4. arm a feature rows (200 random rows): recomputed on CPU with the trained network == the cached row (cosine >= 0.999).
Writes results/audit/align_p1_theirs.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

V = "/workspace/third_party/viviano"
sys.path[:0] = [V, f"{V}/activmask"]
ROOT = Path(__file__).resolve().parents[2]
D = "/workspace/data"
R = f"{D}/replication/viviano"
N = 200
torch.set_num_threads(32)


def image_ids(ds):
    out = []
    for site, k in zip(ds.site, ds.imageids):
        sub = ds.dataset1 if site == 0 else ds.dataset2
        c = sub.csv
        out.append(str(c["Image Index"].iloc[k]) if "Image Index" in c else str(c["ImageID"].iloc[k]) if "ImageID" in c else str(c["patientId"].iloc[k]))
    return out


def raw(task, iid, site):
    import cv2
    p = (f"{D}/cxr_at/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{iid}.jpg" if task == "rsna"
         else f"{D}/cxr_at/{'NIH' if site == 0 else 'PC'}/images-224/{iid}")
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    a = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    h, w = a.shape; c = min(h, w)
    return a[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c].astype(np.float32)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import activmask.utils.configuration as C
    nih = pd.read_csv(f"{D}/cxr_at/NIH/Data_Entry_2017.csv").set_index("Image Index")
    pc = pd.read_csv(f"{D}/cxr_at/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", low_memory=False).drop_duplicates("ImageID").set_index("ImageID")
    rl = pd.read_csv(f"{D}/cxr_at/kaggle-pneumonia-jpg/stage_2_train_labels.csv").groupby("patientId").Target.max()
    rows = []
    rng = np.random.default_rng(20261120)
    for task, cfg_std, arma_cfg, arma_name, arma_mask, ext in (
            ("xray", "resnet", "resnet", "xray_masked_1234", True, None),
            ("rsna", "resnet", "resnet_actdiff", "rsna_arma_lungs_1234", False, f"{R}/arms/rsna_1234")):
        std = pd.read_csv(f"{R}/preds/{task}_standard_1234.csv.gz")
        arm = pd.read_csv(f"{R}/preds/{arma_name}.csv.gz"); F = np.load(f"{R}/preds/{arma_name}_features.npz")
        cfg = C.load_config(f"{V}/activmask/config/{task}-seeds/{task}-seeds_{cfg_std}.yml")
        acfg = C.load_config(f"{V}/activmask/config/{task}-seeds/{task}-seeds_{arma_cfg}.yml")
        run = f"{task}_{arma_cfg}_1234"
        model = torch.load(f"{R}/runs/{run}/results/{task}-seeds_{arma_cfg}/best_model_1234.pth.tar", map_location="cpu", weights_only=False).eval()
        feats = {}
        model.encoder.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o.flatten(1)))
        for split in ("train", "valid", "test"):
            ds = C.setup_dataset(cfg, 1234, split=split)(C.setup_transform(cfg, split))
            ids = image_ids(ds)
            s_std = std[std.split == split].reset_index(drop=True); s_arm = arm[arm.split == split].reset_index(drop=True)
            order_ok = ids == s_std.image_id.astype(str).tolist() == s_arm.image_id.astype(str).tolist()
            pick = rng.choice(len(ids), min(N, len(ids)), replace=False)
            img_bad = lab_bad = site_bad = 0
            for j in pick:
                x = ds[int(j)][0][0].numpy()
                own = raw(task, ids[j], int(ds.site[j]))
                size = x.shape[-1]
                r_own = corr(x, np.asarray(Image.fromarray(own).resize((size, size), Image.BILINEAR), np.float32))
                others = [corr(x, np.asarray(Image.fromarray(raw(task, ids[o], int(ds.site[o]))).resize((size, size), Image.BILINEAR), np.float32))
                          for o in rng.choice(len(ids), 5, replace=False) if o != j]
                img_bad += not (r_own >= 0.97 and r_own > max(others))
                y = int(s_std.y.iloc[j])
                if task == "rsna":
                    lab_bad += y != int(rl[ids[j]])
                    view = ds.dataset1.csv.set_index("patientId") if int(ds.site[j]) == 0 else ds.dataset2.csv.set_index("patientId")
                    site_bad += ids[j] not in view.index
                else:
                    truth = ("Cardiomegaly" in nih.loc[ids[j], "Finding Labels"]) if int(ds.site[j]) == 0 else ("cardiomegaly" in str(pc.loc[ids[j], "Labels"]))
                    lab_bad += y != int(truth)
                    site_bad += (ids[j] in nih.index) != (int(ds.site[j]) == 0)
            # arm a features: recompute with the arm-a network on its own input pipeline (same order, checked above)
            ads = C.setup_dataset(acfg, 1234, split=split)
            dsa = ads(C.setup_transform(acfg, split))
            if arma_mask:
                dsa.mask_all = True
            ext_m = None
            if ext:
                z = np.load(f"{ext}/roi224_{split}.npz"); pos = {k: q for q, k in enumerate(z["ids"].astype(str))}
                ext_m = torch.from_numpy(z["masks"][[pos[i] for i in ids]].astype(np.float32))[:, None]
            fbad = 0
            with torch.no_grad():
                for j in pick:
                    X, seg, _ = dsa[int(j)]
                    X = X[None]
                    if ext_m is not None:
                        X = X * torch.nn.functional.interpolate(ext_m[int(j)][None], size=X.shape[-2:], mode="nearest")
                    model.forward(X, torch.abs(seg[None] - 1).type(torch.BoolTensor))
                    f = feats["f"][0].numpy(); g = F[split][int(j)]
                    fbad += float(f @ g / (np.linalg.norm(f) * np.linalg.norm(g) + 1e-9)) < 0.999
            rows.append({"setting": f"P1 {task}", "split": split, "arm": "their pipeline + arm a features", "rows": len(ids),
                         "order_matches": bool(order_ok), "checked": len(pick), "image_mismatch": img_bad, "label_mismatch": lab_bad,
                         "site_mismatch": site_bad, "feature_mismatch": fbad})
            print(rows[-1], flush=True)
    out = ROOT / "results/audit"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "align_p1_theirs.csv", index=False)


if __name__ == "__main__":
    main()
