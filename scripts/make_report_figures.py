"""Figures for the full project report (report/report.tex): real example images of every cohort with ROI (green) and
artifact (red) overlays, the synthetic artifact families, the generic overlay library, and result charts.

  python scripts/make_report_figures.py      # -> report/figures/*.png
Images are thumbnails of public research datasets (attribution in the report).
"""
from __future__ import annotations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths
from wtss.experiments.real_traps import RealCache

F = paths.ensure(paths.REPO_ROOT / "report" / "figures")
D = paths.DATA


def overlay(img, roi, art, box=False):
    im = (np.repeat(img[..., None], 3, -1) if img.ndim == 2 else img).copy()
    if art is not None:
        m = art > 0
        im[m] = (0.35 * im[m] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cnts, -1, (0, 255, 0), 3)
    return im


def cohort_panel(name, cache, groups, fname, n=4, seed=7):
    """groups: list of (label, list of ids). One row per group, n examples, original | overlay."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(groups), n, figsize=(2.6 * n, 2.7 * len(groups)))
    axes = np.atleast_2d(axes)
    for r, (lab, ids) in enumerate(groups):
        pick = rng.choice(ids, min(n, len(ids)), replace=False) if len(ids) else []
        for c in range(n):
            ax = axes[r, c]; ax.axis("off")
            if c < len(pick):
                img, roi, art = cache.get(pick[c])
                ax.imshow(overlay(np.asarray(img), np.asarray(roi), None if art is None else np.asarray(art)))
            if c == 0:
                ax.set_title(lab, fontsize=9, loc="left")
    fig.suptitle(name, fontsize=11); fig.tight_layout()
    fig.savefig(F / fname, dpi=110); plt.close(fig)


def cohorts():
    # ISIC 2019 hair (Wegley masks)
    c = pd.read_csv(D / "isic2019/prepared/cohort_spec.csv")
    cache = RealCache(D / "isic2019/prepared/cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
    h = c.hair_px_native > 30
    cohort_panel("Dermoscopy (ISIC 2019): hair/ruler (red), lesion (green)", cache,
                 [("hair inside lesion (Trap A)", c[h & (c.r_spec >= 0.5)].image_id.tolist()),
                  ("hair outside lesion (Trap B)", c[h & (c.r_spec < 0.1)].image_id.tolist()),
                  ("hair-free", c[~h].image_id.tolist())], "data_isic.png")
    # thyroid
    t = pd.read_csv(D / "us/tncd/thyroid_cohort.csv")
    cache = RealCache(D / "us/tncd/cache_518", roi_file="roi.npy", art_file="marker.npy")
    p = t.marker_px >= 15
    cohort_panel("Thyroid ultrasound (TN3K): detected calipers (red), nodule (green)", cache,
                 [("calipers inside nodule", t[p & (t.r >= 0.5)].image_id.tolist()),
                  ("calipers outside nodule", t[p & (t.r < 0.1)].image_id.tolist()),
                  ("caliper-free", t[t.marker_px == 0].image_id.tolist())], "data_thyroid.png")
    # ovary
    o = pd.read_csv(D / "ovary/ovary_cohort.csv")
    cache = RealCache(D / "ovary/cache_518", roi_file="roi.npy", art_file="marker.npy")
    p = o.marker_px >= 15
    cohort_panel("Ovarian tumour ultrasound (MMOTU): calipers/overlays (red), tumour (green)", cache,
                 [("calipers inside tumour", o[p & (o.r >= 0.5)].image_id.tolist()),
                  ("overlays outside tumour", o[p & (o.r < 0.1)].image_id.tolist()),
                  ("caliper-free", o[o.marker_px == 0].image_id.tolist())], "data_ovary.png")
    # capsule
    k = pd.read_csv(D / "capsule/capsule_cohort.csv")
    cache = RealCache(D / "capsule/cache_518", roi_file="roi.npy", art_file="contam.npy")
    p = k.contam_frac >= 0.10
    cohort_panel("Capsule endoscopy (SEE-AI): debris (red), lesion box (green)", cache,
                 [("debris on lesion", k[p & (k.r >= 0.5)].image_id.tolist()),
                  ("debris off lesion", k[p & (k.r < 0.1) & (k.roi_cover < 0.05)].image_id.tolist()),
                  ("debris-free", k[k.contam_frac < 0.03].image_id.tolist())], "data_capsule.png")
    # chest drains (no pixel masks: lung ROI only)
    dr = pd.read_csv(D / "cxr/prepared/nih_drain_cohort.csv")
    cache = RealCache(D / "cxr/prepared/cache_nih_drain_518")
    cohort_panel("Chest radiographs (NIH): lung ROI (green); drains are image-level labels", cache,
                 [("pneumothorax with drain", dr[(dr.y == 1) & (dr.a == 1)].image_id.tolist()),
                  ("pneumothorax without drain", dr[(dr.y == 1) & (dr.a == 0)].image_id.tolist())], "data_cxr.png")


def synthetic_and_generic():
    from wtss.synthetic import draw_caliper, draw_debris, draw_generic_artifact, draw_ruler, draw_tube
    t = pd.read_csv(D / "us/tncd/thyroid_cohort.csv"); cache = RealCache(D / "us/tncd/cache_518", roi_file="roi.npy", art_file="marker.npy")
    base_id = t[t.marker_px == 0].image_id.iloc[5]
    g, roi, _ = cache.get(base_id); rgb = Image.fromarray(np.asarray(g))
    k = pd.read_csv(D / "capsule/capsule_cohort.csv"); kc = RealCache(D / "capsule/cache_518", roi_file="roi.npy", art_file="contam.npy")
    kid = k[k.contam_frac < 0.03].image_id.iloc[3]; kimg, kroi, _ = kc.get(kid)
    ys, xs = np.nonzero(np.asarray(roi)); cx, cy = int(xs.mean()), int(ys.mean())
    ic = pd.read_csv(D / "isic2018/isic2018_pilot_manifest.csv") if (D / "isic2018/isic2018_pilot_manifest.csv").exists() else None
    isc = RealCache(D / "isic2019/prepared/cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
    ci = pd.read_csv(D / "isic2019/prepared/cohort_spec.csv"); iid = ci[ci.hair_px_native <= 30].image_id.iloc[11]
    iimg, iroi, _ = isc.get(iid); ys, xs = np.nonzero(np.asarray(iroi)); ix, iy = int(xs.mean()), int(ys.mean())
    xc = RealCache(D / "cxr/prepared/cache_nih_ptx_518", roi_file="roi.npy") if (D / "cxr/prepared/cache_nih_ptx_518").exists() else None
    if xc is not None:
        xid = xc.ids[5]; ximg, xroi, _ = xc.get(xid); ys2, xs2 = np.nonzero(np.asarray(xroi)); tx, ty = int(np.median(xs2)), int(np.median(ys2))
        tube = draw_tube(Image.fromarray(np.asarray(ximg)), tx - 50, ty - 6, 100, 12, "demo")[0]
    else:
        tube = draw_tube(rgb, cx - 50, cy - 6, 100, 12, "demo")[0]
    tiles = [("ruler (dermoscopy)", draw_ruler(Image.fromarray(np.asarray(iimg)), ix - 32, iy - 9, 65, 19)[0]),
             ("tube (chest X-ray)", tube),
             ("caliper (ultrasound)", draw_caliper(rgb, cx - 32, cy - 9, 65, 19, "demo")[0]),
             ("debris (capsule)", draw_debris(Image.fromarray(np.asarray(kimg)), 230, 230, 48, 48, "demo")[0])]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (lab, im) in zip(axes, tiles):
        ax.imshow(im); ax.set_title(lab, fontsize=9); ax.axis("off")
    fig.suptitle("Controlled synthetic artifacts (placed at chosen overlap with the ROI)"); fig.tight_layout()
    fig.savefig(F / "synthetic_artifacts.png", dpi=110); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(draw_generic_artifact(rgb if i < 4 else Image.fromarray(np.asarray(kimg)),
                                        np.asarray(roi) if i < 4 else np.asarray(kroi), f"gen{i}")); ax.axis("off")
    fig.suptitle("Generic overlay library used by U-MtE (never shown a real artifact)"); fig.tight_layout()
    fig.savefig(F / "generic_overlays.png", dpi=110); plt.close(fig)


def charts():
    # crossover per cohort (primary P1) and U-MtE gains (from PRIMARY_CLAIMS)
    d = pd.read_csv(paths.RESULTS / "PRIMARY_CLAIMS.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    for ax, fam in zip(axes, sorted(d.family.unique())):
        q = d[d.family == fam]
        ok = (q.p_holm < 0.05) & (q.estimate > 0)
        ax.bar(q.cohort.str.replace(r" \(DINOv2\)", "", regex=True), q.estimate, yerr=[q.estimate - q.ci95_lo, q.ci95_hi - q.estimate],
               color=np.where(ok, "tab:blue", "tab:red"), capsize=3)
        ax.axhline(0, color="k", lw=.6); ax.set_title(fam.split(" ", 1)[1], fontsize=9); ax.tick_params(axis="x", labelsize=7, rotation=20)
    axes[0].set_ylabel("Δ reversed AUROC")
    fig.tight_layout(); fig.savefig(F / "primary_bars.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    cohorts(); charts(); print("report figures ->", F)  # report shows real images only (user request)
