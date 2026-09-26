"""Outcome-free feasibility of TCGA LUAD vs LUSC artifact traps from GrandQC QC masks (MPP 1.5).

Per slide: tissue fraction; per artifact class: area relative to tissue and overlap r = fraction of the
artifact lying on tissue (tissue ROI = closed/filled union of tissue + artifact-on-tissue pixels).
"""
import glob
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths

Image.MAX_IMAGE_PIXELS = None
CLS = {2: "fold", 3: "dark_foreign", 4: "pen", 5: "bubble_edge", 6: "out_of_focus"}


def one(f):
    a = np.asarray(Image.open(f))[::4, ::4]  # 6 µm/px is plenty for area statistics
    tissue = (a == 1)
    # ROI = tissue region with holes/artifacts inside it filled (closing + hole filling)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(closed)
    cv2.drawContours(roi, cnts, -1, 1, thickness=-1)
    roi = roi.astype(bool)
    row = {"file": f.split("/")[-1], "cohort": f.split("/")[-3].split("_")[0], "tissue_px": int(tissue.sum()),
           "roi_px": int(roi.sum())}
    for c, n in CLS.items():
        m = a == c
        row[f"{n}_px"] = int(m.sum())
        row[f"{n}_r"] = float((m & roi).sum() / m.sum()) if m.sum() else np.nan
    return row


if __name__ == "__main__":
    fs = sorted(glob.glob(str(paths.DATA / "path" / "grandqc" / "*" / "mask_qc" / "*.png")))
    with ProcessPoolExecutor(6) as ex:
        d = pd.DataFrame(list(ex.map(one, fs, chunksize=8)))
    d["slide"] = d.file.str.replace(".svs_mask.png", "", regex=False)
    d["case"] = d.slide.str[:12]
    d["site"] = d.slide.str.extract(r"TCGA-(\w\w)-")[0]
    d["y"] = (d.cohort == "LUSC").astype(int)
    for n in CLS.values():
        d[f"{n}_frac"] = d[f"{n}_px"] / d.roi_px.clip(lower=1)
    out = paths.ensure(paths.DATA / "path" / "prepared")
    d.to_csv(out / "grandqc_lung_stats.csv", index=False)
    print(d.cohort.value_counts().to_dict(), "slides;", d.case.nunique(), "cases;", d.site.nunique(), "sites")
    for n in CLS.values():
        pres = d[f"{n}_frac"] > 0.001
        print(f"{n:13s} present(>0.1% of tissue): {pres.mean():.2f}  median r (on tissue) {d.loc[pres, f'{n}_r'].median():.2f}  "
              f"r>=0.5: {(pres & (d[f'{n}_r'] >= 0.5)).sum():4d}  r<0.1: {(pres & (d[f'{n}_r'] < 0.1)).sum():4d}  "
              f"P(present|LUAD)={pres[d.y == 0].mean():.2f} P(present|LUSC)={pres[d.y == 1].mean():.2f}")
    clean = (d[[f"{n}_frac" for n in CLS.values()]] <= 0.001).all(1)
    print("artifact-free slides (all classes <= 0.1% of tissue):", int(clean.sum()), "LUAD", int((clean & (d.y == 0)).sum()), "LUSC", int((clean & (d.y == 1)).sum()))
