"""External validation on ISIC 2020 (patient-level identifiers), docs/PREREGISTRATION_FINAL.md A4.

Stages (each resumable):
 1. hair segmenter: U-Net (ImageNet ResNet-34 encoder) trained on ISIC 2019 518-px images and their hair masks only; the
    image-level hair-free threshold tau (predicted hair pixels at 518 px) is calibrated on held-out ISIC 2019 images
    against the paper's hair-free definition (<= 30 native hair pixels). Frozen before any ISIC 2020 image is scored.
 2. ISIC 2020 cache (518 px), lesion masks from the existing spec U-Net (unet_spec_256.pt), hair masks from stage 1.
 3. Cell-count gate (pre-registered): both traps need >= 50 positive and >= 50 negative artifact-bearing images and
    >= 50 positive and >= 50 negative hair-free images. If the gate fails, stop and record why.
 4. Only if the gate holds: spec traps (erm, mask; DINOv2@518; 5 seeds x 5 folds; groups = patient_id).
Outputs: <WTSS_RESULTS>/external_isic2020/{segmenter.json, gate.json, cohort.csv, ...}
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths

Image.MAX_IMAGE_PIXELS = None
E = paths.DATA / "external" / "isic2020"
OUT = paths.ensure(paths.RESULTS / "external_isic2020")
W = paths.ensure(paths.DATA / "external" / "isic2020_work")
P19 = paths.DATA / "isic2019" / "prepared"
S = 518
GATE = 50


def log(**kw):
    print(json.dumps(kw), flush=True)


def seg_model():
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)


def norm(x, dev):
    t = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255
    m = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return (t - m) / s


def stage1():
    f = OUT / "segmenter.json"
    if f.exists() and (W / "hair_unet.pt").exists():
        return json.loads(f.read_text())
    c = pd.read_csv(P19 / "cohort_spec.csv")
    ids = (P19 / "cache_518" / "ids.txt").read_text().split()
    pos = {k: j for j, k in enumerate(ids)}
    rgb = np.load(P19 / "cache_518" / "rgb.npy", mmap_mode="r")
    hair = np.load(P19 / "cache_518" / "hair.npy", mmap_mode="r")
    c = c[c.image_id.isin(pos)].reset_index(drop=True)
    rng = np.random.default_rng(20260928)
    val = rng.random(len(c)) < 0.1
    tr_idx = np.array([pos[i] for i in c.image_id[~val]]); va = c[val].reset_index(drop=True)
    dev = torch.device("cuda")
    net = seg_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4)
    scaler = torch.amp.GradScaler()
    bs, steps = 12, 3000
    for k in range(steps):
        b = tr_idx[rng.integers(0, len(tr_idx), bs)]
        x = np.stack([rgb[j] for j in b]); y = np.stack([hair[j] for j in b]).astype(np.float32)
        if rng.random() < .5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        with torch.autocast("cuda", dtype=torch.float16):
            lo = net(norm(np.ascontiguousarray(x), dev))
        yt = torch.from_numpy(np.ascontiguousarray(y)).to(dev)[:, None]
        p = torch.sigmoid(lo.float())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), yt) + \
            (1 - (2 * (p * yt).sum((2, 3)) + 1) / (p.sum((2, 3)) + yt.sum((2, 3)) + 1)).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if k % 500 == 0:
            log(stage="seg_train", step=k, loss=float(loss))
    torch.save(net.state_dict(), W / "hair_unet.pt")
    net.eval()
    pred_px, ious = [], []
    with torch.inference_mode():
        for k0 in range(0, len(va), 32):
            q = va.iloc[k0:k0 + 32]
            x = np.stack([rgb[pos[i]] for i in q.image_id])
            with torch.autocast("cuda", dtype=torch.float16):
                pm = (torch.sigmoid(net(norm(x, dev))) > 0.5).cpu().numpy()[:, 0]
            for m, i in zip(pm, q.image_id):
                g = hair[pos[i]] > 0
                pred_px.append(int(m.sum()))
                u = (m | g).sum()
                ious.append(float((m & g).sum() / u) if u else 1.0)
    va["pred_px"] = pred_px
    truth = (va.hair_px_native <= 30).to_numpy()
    best = max(((t, ((va.pred_px <= t) == truth).mean()) for t in (0, 5, 10, 20, 50, 100, 200, 400, 800)), key=lambda z: z[1])
    ba = lambda t: 0.5 * (((va.pred_px <= t) & truth).sum() / max(truth.sum(), 1) + ((va.pred_px > t) & ~truth).sum() / max((~truth).sum(), 1))
    tau = max((0, 5, 10, 20, 50, 100, 200, 400, 800), key=ba)
    rep = {"n_val": int(len(va)), "pixel_iou_mean": float(np.mean(ious)), "tau_pred_hair_px": int(tau),
           "hair_free_balanced_accuracy": float(ba(tau)), "hair_free_rate_true": float(truth.mean()),
           "hair_free_rate_pred": float((va.pred_px <= tau).mean()), "accuracy_best_tau": float(best[1])}
    f.write_text(json.dumps(rep, indent=1))
    log(stage="segmenter", **rep)
    return rep


def stage2(rep):
    coh = OUT / "cohort.csv"
    if coh.exists():
        return pd.read_csv(coh)
    gt = pd.read_csv(E / "ISIC_2020_Training_GroundTruth_v2.csv")
    cdir = paths.ensure(W / "cache_518")
    if (cdir / "ids.txt").exists() and (cdir / "rgb.npy").exists():  # RGB cache complete (ids.txt is written last)
        ids = (cdir / "ids.txt").read_text().split()
        gt = gt.set_index("image_name").loc[ids].reset_index()
        rgb = np.load(cdir / "rgb.npy", mmap_mode="r")
        n = len(gt)
        assert rgb.shape[0] == n
    else:
        img_dir = E / "train"
        if not img_dir.exists():
            with zipfile.ZipFile(E / "ISIC_2020_Training_JPEG.zip") as z:
                z.extractall(E)
        files = {p.stem: p for p in E.rglob("*.jpg")}
        gt = gt[gt.image_name.isin(files)].reset_index(drop=True)
        n = len(gt)
        rgb = np.lib.format.open_memmap(cdir / "rgb.npy", "w+", np.uint8, (n, S, S, 3))

        def load(i):
            with Image.open(files[i]) as im:
                im.draft("RGB", (S * 2, S * 2))
                return np.asarray(im.convert("RGB").resize((S, S), Image.BICUBIC))
        with ThreadPoolExecutor(24) as ex:
            for k, a in enumerate(ex.map(load, gt.image_name)):
                rgb[k] = a
        rgb.flush()
        (cdir / "ids.txt").write_text("\n".join(gt.image_name))
    log(stage="cache", n=n)
    # disk guard: the two uint8 mask arrays need 2 * n * S * S bytes; stop cleanly instead of filling the disk
    import shutil
    need = sum(n * S * S - int(os.stat(cdir / f).st_blocks) * 512 if (cdir / f).exists() else n * S * S
               for f in ("roi.npy", "hair.npy"))
    free = shutil.disk_usage(cdir).free
    log(stage="disk", need_gb=need / 1e9, free_gb=free / 1e9)
    if free - need < 8e9:
        raise SystemExit(f"not enough disk for the mask arrays: need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    dev = torch.device("cuda")
    sys.path.insert(0, str(paths.REPO_ROOT / "scripts" / "data"))
    from train_unet_spec import UNet  # existing lesion U-Net (spec), trained on ISIC 2018 Task 1 only
    les = UNet().to(dev); les.load_state_dict(torch.load(paths.DATA / "isic2019" / "unet_spec" / "unet_spec_256.pt", map_location=dev)); les.eval()
    hnet = seg_model().to(dev); hnet.load_state_dict(torch.load(W / "hair_unet.pt", map_location=dev)); hnet.eval()
    def mm(f):  # reuse the (partly written) arrays of an interrupted run; every row is recomputed below
        p = cdir / f
        if p.exists():
            a = np.load(p, mmap_mode="r+")
            if a.shape == (n, S, S):
                return a
        return np.lib.format.open_memmap(p, "w+", np.uint8, (n, S, S))
    roi, hm = mm("roi.npy"), mm("hair.npy")
    stats = []
    with torch.inference_mode():
        for k0 in range(0, n, 32):
            x = np.asarray(rgb[k0:k0 + 32])
            x256 = np.stack([np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR)) for a in x])
            with torch.autocast("cuda", dtype=torch.float16):
                lm = torch.sigmoid(les(norm(x256, dev))).float()
                lm = torch.nn.functional.interpolate(lm, size=(S, S), mode="nearest")[:, 0].cpu().numpy() > 0.5
                hmk = (torch.sigmoid(hnet(norm(x, dev))).float() > 0.5)[:, 0].cpu().numpy()
            roi[k0:k0 + len(x)] = lm; hm[k0:k0 + len(x)] = hmk
            for a, b in zip(lm, hmk):
                n_h = int(b.sum())
                stats.append({"lesion_frac": float(a.mean()), "hair_px": n_h, "r": float((a & b).sum() / n_h) if n_h else np.nan})
    roi.flush(); hm.flush()
    d = pd.concat([gt, pd.DataFrame(stats)], axis=1)
    d.to_csv(coh, index=False)
    return d


def stage3(d, rep):
    tau = rep["tau_pred_hair_px"]
    d = d[d.lesion_frac > 0].copy()
    d["A0"] = d.hair_px <= tau
    d["trapA_A1"] = ~d.A0 & (d.r >= 0.5)
    d["trapB_A1"] = ~d.A0 & (d.r < 0.1)
    cells = {"A0_pos": int((d.A0 & (d.target == 1)).sum()), "A0_neg": int((d.A0 & (d.target == 0)).sum()),
             "trapA_A1_pos": int((d.trapA_A1 & (d.target == 1)).sum()), "trapA_A1_neg": int((d.trapA_A1 & (d.target == 0)).sum()),
             "trapB_A1_pos": int((d.trapB_A1 & (d.target == 1)).sum()), "trapB_A1_neg": int((d.trapB_A1 & (d.target == 0)).sum())}
    ok = all(v >= GATE for v in cells.values())
    g = {"gate_min_per_cell": GATE, **cells, "gate_met": ok, "n_images": int(len(d)), "n_patients": int(d.patient_id.nunique()),
         "n_melanoma": int(d.target.sum())}
    (OUT / "gate.json").write_text(json.dumps(g, indent=1))
    log(stage="gate", **g)
    return d, ok


def stage4(d):
    from wtss.data.isic2019_spec import build_spec_envs
    from wtss.experiments.real_traps import RealCache
    from wtss.experiments.spec_traps import run_spec
    from wtss.stats import difference_of_deltas
    d = d.rename(columns={"image_name": "image_id", "target": "y"})
    d["source"], d["lesion_id"], d["group"] = "ISIC2020", d.image_id, d.patient_id.astype(str)
    envs = build_spec_envs(d, group_col="group")
    cache = RealCache(W / "cache_518", roi_file="roi.npy", art_file="hair.npy")
    out = paths.ensure(OUT / "traps_dino518")
    if not (out / "predictions.csv.gz").exists():
        run_spec(envs, cache, "dino518", out, paths.CACHE / "features" / "isic2020" / "dinov2_b14_518", [],
                 arms=("erm", "mask"), device=torch.device("cuda"))
    p = pd.read_csv(out / "predictions.csv.gz")
    x = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
    (out / "X1_crossover.json").write_text(json.dumps(x, indent=1, default=float))
    log(stage="X1", crossover=x["seed_delta_mean"], lo=x["ci95_lo"], hi=x["ci95_hi"])


if __name__ == "__main__":
    rep = stage1()
    d = stage2(rep)
    d, ok = stage3(d, rep)
    if ok:
        stage4(d)
    else:
        log(stage="stop", reason="pre-registered cell-count gate not met; external validation not run")
