"""CXR real-device cohort: RANZCR-CLiP images linked to NIH ChestX-ray14 labels.

Outputs ($WTSS_DATA/cxr/prepared/):
  clip_cohort.csv       one row per linked image: NIH findings, patient, view, CLiP device labels,
                        per-device overlap r_<dev> = |device ∩ lungs| / |device| (annotated images only)
  cache_clip_518/       gray.npy, roi.npy (lungs, PSPNet), dev.npy (bit mask: 1 ETT, 2 NGT, 4 CVC, 8 Swan-Ganz)

Device masks are the human CLiP polylines rasterised at 518 px with thickness from typical tube calibre
(chest width ≈ 350 mm ≈ 518 px): ETT 12 px (~8 mm), NGT 7 px (~5 mm), CVC and Swan-Ganz 4 px (~3 mm).
"""
import ast
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths

sys.path.insert(0, str(paths.REPO_ROOT / "scripts" / "data"))
from prepare_cxr import LungSeg  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
S = 518
CLIP = paths.DATA / "cxr" / "clip"
PREP = paths.DATA / "cxr" / "prepared"
BIT = {"ETT": 1, "NGT": 2, "CVC": 4, "Swan": 8}
THICK = {"ETT": 12, "NGT": 7, "CVC": 4, "Swan": 4}


def load(uid):
    with Image.open(CLIP / "train" / f"{uid}.jpg") as im:
        W, H = im.size
        im.draft("L", (S * 2, S * 2))
        g = np.asarray(im.convert("L").resize((S, S), Image.BICUBIC), np.uint8)
    return g, W, H


def main():
    t = pd.read_csv(CLIP / "train.csv")
    link = pd.read_csv(PREP / "clip_nih_link.csv")
    link = link[link.accepted]
    nih = pd.read_csv(paths.CXR_NIH / "data" / "Data_Entry_2017_v2020.csv").rename(
        columns={"Image Index": "nih_image", "Finding Labels": "findings", "Patient ID": "nih_patient",
                 "View Position": "view", "Patient Age": "age", "Patient Gender": "sex"})
    c = t.merge(link[["StudyInstanceUID", "nih_image"]], on="StudyInstanceUID").merge(
        nih[["nih_image", "findings", "nih_patient", "view", "age", "sex"]], on="nih_image")
    for d in BIT:
        c[f"has_{d}"] = c[[x for x in t.columns if x.startswith(d)]].sum(axis=1).gt(0).astype(int)
    ann = pd.read_csv(CLIP / "train_annotations.csv")
    ann["dev"] = ann.label.str.split(" ").str[0].replace({"Swan": "Swan"})
    by_uid = {u: g for u, g in ann.groupby("StudyInstanceUID")}
    uids = c.StudyInstanceUID.tolist()
    cdir = PREP / f"cache_clip_{S}"
    cdir.mkdir(parents=True, exist_ok=True)
    n = len(uids)
    gray = np.lib.format.open_memmap(cdir / "gray.npy", "w+", np.uint8, (n, S, S))  # CXR: 1 channel
    roi = np.lib.format.open_memmap(cdir / "roi.npy", "w+", np.uint8, (n, S, S))
    dev = np.lib.format.open_memmap(cdir / "dev.npy", "w+", np.uint8, (n, S, S))
    seg = LungSeg(torch.device("cuda"))
    r = {d: np.full(n, np.nan) for d in BIT}
    area = {d: np.zeros(n) for d in BIT}
    annotated = np.zeros(n, bool)
    with ThreadPoolExecutor(8) as ex:
        for k0 in range(0, n, 64):
            chunk = uids[k0:k0 + 64]
            loaded = list(ex.map(load, chunk))
            g512 = np.stack([np.asarray(Image.fromarray(g).resize((512, 512), Image.BILINEAR)) for g, _, _ in loaded])
            lungs = seg(g512)
            for j, (uid, (g, W, H), lm) in enumerate(zip(chunk, loaded, lungs)):
                k = k0 + j
                gray[k] = g
                L = np.asarray(Image.fromarray(lm * 255).resize((S, S), Image.NEAREST)) >= 128
                roi[k] = L
                m = np.zeros((S, S), np.uint8)
                if uid in by_uid:
                    annotated[k] = True
                    for _, row in by_uid[uid].iterrows():
                        pts = np.array(ast.literal_eval(row.data), float)
                        if len(pts) < 2:
                            continue
                        pts = (pts * [S / W, S / H]).astype(np.int32)
                        one = np.zeros((S, S), np.uint8)
                        cv2.polylines(one, [pts], False, 1, thickness=THICK[row.dev], lineType=cv2.LINE_8)
                        m |= one * BIT[row.dev]
                    for d, b in BIT.items():
                        md = (m & b) > 0
                        if md.any():
                            area[d][k] = md.mean()
                            r[d][k] = (md & L).sum() / md.sum()
                dev[k] = m
            if k0 % 3200 == 0:
                print(f"[clip] {k0 + len(chunk)}/{n}", flush=True)
    gray.flush(); roi.flush(); dev.flush()
    (cdir / "ids.txt").write_text("\n".join(uids))
    c["annotated"] = annotated
    c["lung_frac"] = [float(np.asarray(roi[k]).mean()) for k in range(n)]
    for d in BIT:
        c[f"r_{d}"] = r[d]
        c[f"area_{d}"] = area[d]
    c.to_csv(PREP / "clip_cohort.csv", index=False)
    print(c[[f"r_{d}" for d in BIT]].describe().round(3))


if __name__ == "__main__":
    main()
