"""P1 location test (amendment A15) in P1's own pipeline: their training code and Standard config, unchanged, on P1's
No-SPC split (ratio 0.5, seed 1234) with our radio-opaque tube inside or outside the lungs + heart ROI.

Runs in P1's environment (/workspace/envs/viviano). Their `training.train` and `JointDataset` are used as released (at
32f7b7d + our port patch); a subclass draws the tube between their intensity normalisation and their resize to 128 px,
and `configuration.setup_dataset` is redirected to it at runtime (no file of theirs is edited).

    python scripts/replication/p1_location.py train --position inside|outside        # trains Standard (seed 1234)
    python scripts/replication/p1_location.py score --position inside|outside        # Standard + Masked (A5) scores
    python scripts/replication/p1_location.py score --position ... --model last      # A17 (post hoc): their epoch-100 model
Outputs: $WTSS_DATA/replication/viviano/location/<position>/ (run dir with their results/, scores_*.csv.gz).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V = "/workspace/third_party/viviano"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [V, f"{V}/activmask", str(ROOT / "src")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
SEED = int(os.environ.get("P1_SEED", "1234"))   # A43: seeds 3232 / 3221 added; 1234 keeps its original paths
L = Path(f"{D}/replication/viviano/location" + ("" if SEED == 1234 else f"_s{SEED}"))
W, H = 43, 6                                   # TUBE_GEOMETRY[224]
P_TUBE = {"train": (0.1, 0.9), "valid": (0.9, 0.1), "corr": (0.1, 0.9), "rev": (0.9, 0.1), "clean": (0.0, 0.0)}  # P(tube | y=0), P(tube | y=1)
PATHS = dict(d1data=f"{D}/cxr_at/NIH/images-224", d1csv=f"{D}/cxr_at/NIH/Data_Entry_2017.csv",
             d2data=f"{D}/cxr_at/PC/images-224", d2csv=f"{D}/cxr_at/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")


def make_dataset_class():
    import torchvision.transforms.functional as TF
    import activmask.datasets.xray as X
    from wtss.synthetic import tube_masks
    from wtss.utils import stable_int

    pl = pd.read_csv(L / f"placements_{SEED}.csv").set_index("key")

    class TubeJointDataset(X.JointDataset):
        def __init__(self, position="inside", env=None, **kw):
            super().__init__(**kw)
            # inside their training loop the test split is scored as the reversed test (P1's design)
            self.position, self.env = position, env or {"test": "rev"}.get(kw.get("mode"), kw.get("mode"))
            sub = lambda s: self.dataset1 if s == 0 else self.dataset2
            self.keys = [f"{s}/" + (str(sub(s).csv["Image Index"].iloc[i]) if s == 0 else str(sub(s).csv["ImageID"].iloc[i]))
                         for s, i in zip(self.site, self.imageids)]
            p0, p1 = P_TUBE[self.env]
            self.tube = np.array([stable_int("p1loc", k, int(y), SEED, self.env) / (2**32 - 1) < (p1 if y == 1 else p0)
                                  for k, y in zip(self.keys, self.labels)])

        def __getitem__(self, idx):   # their JointDataset.__getitem__, plus the tube before convert/resize (A15)
            dataset = self.dataset1 if self.site[idx] == 0 else self.dataset2
            img, _, _ = dataset[self.imageids[idx]]
            seg = self.seg[None, :, :]
            a = img[0, :, :].astype(np.float32)
            if self.tube[idx]:
                r = pl.loc[self.keys[idx]]
                x, y = (int(r.in_x), int(r.in_y)) if self.position == "inside" else (int(r.out_x), int(r.out_y))
                m, stripe = tube_masks(a.shape, x, y, W, H, self.keys[idx])
                bright = np.clip(a + 70 / 255 * 2048, -1024, 1024)           # draw_tube's +70 of 255, on their scale
                a = np.where(m == 1, 0.35 * a + 0.65 * bright, a)
                a = np.where(stripe == 1, (2 * 245 / 255 - 1) * 1024, a)       # draw_tube's stripe value 245
            img = self.convert(a)
            if self.new_size != img.width == img.height:
                img = self.resize(img)
            img = TF.to_tensor(img).float()
            seg = TF.to_tensor(seg).permute([1, 0, 2]).float()
            if self.mask_all:
                img *= seg
            return (img, seg, self.labels[idx])

    return TubeJointDataset


def config(position):
    import activmask.utils.configuration as C
    cfg = C.load_config(f"{V}/activmask/config/xray-seeds/xray-seeds_resnet.yml")   # their Standard config
    for split in ("train", "valid", "test"):
        cfg["dataset"][split] = {"TubeJointDataset": {**PATHS, "ratio": 0.5, "mode": split, "position": position}}
    cfg["experiment_name"] = f"xray-location_{position}"
    return cfg


def redirect(cls):
    import activmask.utils.configuration as C
    def setup_dataset(config, seed, split="train"):
        args = dict(list(config["dataset"][split].values())[0])
        return lambda transform: cls(transform=transform, seed=seed, **args)
    C.setup_dataset = setup_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "train", "score"])
    ap.add_argument("--position", choices=["inside", "outside"], required=True)
    ap.add_argument("--model", choices=["best", "last"], default="best")   # last: A17 post-hoc sensitivity analysis
    args = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.stage == "list":   # their JointDataset split listing (ratio 0.5, this seed), read by p1_location_prep.py
        import activmask.datasets.xray as X
        L.mkdir(parents=True, exist_ok=True); rows = []
        for split in ("train", "valid", "test"):
            ds = X.JointDataset(mode=split, seed=SEED, ratio=0.5, verbose=False, **PATHS)
            for pos, (st, i) in enumerate(zip(ds.site, ds.imageids)):
                if st == 0:
                    iid = str(ds.dataset1.csv["Image Index"].iloc[i]); path = f"NIH/images-224/{iid}"
                else:
                    iid = str(ds.dataset2.csv["ImageID"].iloc[i]); path = f"PC/images-224/{iid}"
                rows.append({"site": int(st), "image_id": iid, "path": path, "split": split, "pos": pos, "y": int(ds.labels[pos])})
        pd.DataFrame(rows).to_csv(L / f"nospc_{SEED}.csv", index=False); print("listed", len(rows), "rows for seed", SEED)
        return
    out = L / args.position; out.mkdir(parents=True, exist_ok=True); os.chdir(out)
    cls = make_dataset_class()
    if args.stage == "train":
        import training
        redirect(cls)
        training.train(config(args.position), random_state=SEED)
        return
    import torch
    from sklearn.metrics import roc_auc_score
    torch.cuda.set_per_process_memory_fraction(0.15)
    sfx = "" if args.model == "best" else "_last"
    model = torch.load(out / f"results/xray-location_{args.position}/{args.model}_model_{SEED}.pth.tar", map_location="cuda",
                       weights_only=False).eval()
    feats = {}
    model.encoder.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o.flatten(1).cpu()))
    rows, F = [], {}
    for env in ("train", "valid", "corr", "rev", "clean"):
        split = env if env in ("train", "valid") else "test"
        for method, mask in (("standard", False), ("masked", True)):
            if env in ("train", "valid") and method == "standard":
                continue          # train/valid features are needed only for arm a (masked input, every split)
            ds = cls(position=args.position, env=env, mode=split, seed=SEED, ratio=0.5, mask_all=mask, verbose=False, **PATHS)
            dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=12)
            lg, fs = [], []
            with torch.no_grad():
                for X_, seg, y in dl:
                    seg = torch.abs(seg - 1).type(torch.BoolTensor).cuda()
                    lg.append(model.forward(X_.cuda(), seg)["y_pred"].cpu()); fs.append(feats["f"])
            L_ = torch.cat(lg).numpy()
            d = pd.DataFrame({"image_id": ds.keys, "label": ds.labels, "tube": ds.tube.astype(int), "method": method,
                              "test_set": env, "logit0": L_[:, 0], "logit1": L_[:, 1], "score": L_[:, 1] - L_[:, 0]})
            rows.append(d)
            if method == "masked":
                F[env] = torch.cat(fs).numpy()
            if split == "test":
                print(args.position, method, env, "AUROC", round(roc_auc_score(d.label, d.score), 4), flush=True)
    pd.concat(rows).to_csv(out / f"scores{sfx}.csv.gz", index=False)
    np.savez_compressed(out / f"features_masked{sfx}.npz", **F)


if __name__ == "__main__":
    main()
