"""Mechanism analyses with the corrected (crossed) bootstrap, from the regenerated sweeps (docs/PREREGISTRATION_FINAL.md, A1).

  -> <NEW>/final_mechanism/lesion_tertiles.csv      E7: mask - ERM (reversed AUROC) by lesion-size tertile and overlap
     <NEW>/final_mechanism/counterfactual_ci.csv    E6: same-head |dp| of mask and ERM and their difference, per sweep/overlap
     <NEW>/final_mechanism/dilation_retention.csv   E5: share of ruler pixels visible after masking with each dilation margin
Skips a part whose input is missing.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

os.environ["WTSS_BOOTSTRAP"] = "crossed"
from wtss import paths  # noqa: E402
from wtss import stats_crossed as X  # noqa: E402
from wtss.data.isic2018 import load_isic2018_pilot  # noqa: E402
from wtss.synthetic import ARTIFACT_GEOMETRY  # noqa: E402

R = paths.RESULTS
OUT = paths.ensure(R / "final_mechanism")
SWEEPS = {"dermoscopy ruler (DINOv2)": "synthetic/isic2018/dino518_ruler_fixed_corr_main",
          "thyroid caliper (DINOv2)": "synthetic/thyroid/dino518_caliper_corr_main",
          "ovary caliper (DINOv2)": "synthetic/ovary/dino518_caliper_corr_main",
          "capsule debris (DINOv2)": "synthetic/capsule/dino518_debris_corr_main",
          "chest tube (RAD-DINO)": "synthetic/nih_ptx/raddino518_tube_corr_main"}


def tertiles():
    f = R / SWEEPS["dermoscopy ruler (DINOv2)"] / "predictions.csv.gz"
    if not f.exists():
        return
    coh = load_isic2018_pilot().df
    cdir = paths.CACHE / "images" / "isic2018_pilot_518"
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    ids = (cdir / "ids.txt").read_text().split()
    area = pd.Series([float(np.asarray(roi[j]).mean()) for j in range(len(ids))], index=ids)
    p = pd.read_csv(f)
    test_ids = coh[coh.split == "test"].image_id
    tert = pd.qcut(area[test_ids], 3, labels=["small", "medium", "large"])
    w, h = ARTIFACT_GEOMETRY[518]
    rows = []
    for ov in (0.0, 0.5, 1.0):
        for t in ("large", "medium", "small"):
            keep = set(tert[tert == t].index)
            q = p[np.isclose(p.overlap, ov) & p.image_id.isin(keep)]
            r = X.hierarchical_paired_bootstrap(q, "mask", "erm", "test_rev", 10000, 20260928 + int(ov * 100))
            rows.append({"overlap": ov, "tertile": t, "n_melanoma": int(q[(q.method == "erm") & (q.env == "test_rev") & (q.seed == 42)].y.sum()),
                         "seed_delta_mean": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
                         "p_boot_two_sided": r["p_boot_two_sided"],
                         "artifact_to_lesion_ratio_median": float(np.median(w * h / (area[list(keep)] * 518 * 518)))})
    pd.DataFrame(rows).to_csv(OUT / "lesion_tertiles.csv", index=False)
    print("tertiles done", flush=True)


def counterfactual():
    rows = []
    for name, rel in SWEEPS.items():
        f = R / rel / "counterfactual_per_image.csv.gz"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        for ov in sorted(c.overlap.unique()):
            q = c[np.isclose(c.overlap, ov) & c.method.isin(["erm", "mask"])]
            w = q.pivot_table(index=["image_id", "seed"], columns="method", values="abs_delta_p").dropna().reset_index()
            if w.empty:
                continue
            r = X.hierarchical_paired_mean_bootstrap(w, "mask", "erm", 10000, 20260928 + int(ov * 100))
            rows.append({"sweep": name, "overlap": ov, "absdp_mask": float(w["mask"].mean()), "absdp_erm": float(w["erm"].mean()),
                         "seed_delta_mean": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
                         "p_boot_two_sided": r["p_boot_two_sided"], "n_pairs": len(w)})
        print(name, "counterfactual done", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "counterfactual_ci.csv", index=False)


def retention():
    pl = json.loads((paths.FROZEN / "placements_518_65x19.json").read_text())
    coh = load_isic2018_pilot().df
    cdir = paths.CACHE / "images" / "isic2018_pilot_518"
    if not (cdir / "roi.npy").exists():
        return
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    idx = {k: j for j, k in enumerate((cdir / "ids.txt").read_text().split())}
    w, h = ARTIFACT_GEOMETRY[518]
    rows = []
    for ov in ("0.50", "0.75", "1.00"):
        for m in (0, 10, 25, 50):
            ret, share = [], []
            for i in coh[coh.split == "test"].image_id:
                if i not in pl or i not in idx:
                    continue
                r = (np.asarray(roi[idx[i]]) > 0).astype(np.uint8)
                if m:
                    r = cv2.dilate(r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1)))
                p = pl[i][ov]
                box = np.zeros_like(r); box[p["y"]:p["y"] + h, p["x"]:p["x"] + w] = 1
                vis = r > 0
                ret.append((box & r).sum() / box.sum())
                share.append((box & r).sum() / max(vis.sum(), 1))
            rows.append({"overlap": float(ov), "margin_px": m, "retention_mean": float(np.mean(ret)),
                         "ruler_share_of_visible_mean": float(np.mean(share)), "n_images": len(ret)})
    pd.DataFrame(rows).to_csv(OUT / "dilation_retention.csv", index=False)
    print("retention done", flush=True)


if __name__ == "__main__":
    tertiles(); counterfactual(); retention()
