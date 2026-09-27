"""Paper figures, generated from results/ (never edited by hand).

  python scripts/make_figures.py   # -> paper/figures/{dose_response,forest,examples}.pdf/.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths

R, F = paths.RESULTS, paths.ensure(paths.REPO_ROOT / "paper" / "figures")
SWEEPS = [("Dermoscopy ruler · DINOv2", "synthetic/isic2018/dino518_ruler_fixed_corr_main"),
          ("Dermoscopy ruler · DermLIP", "synthetic/isic2018/dermlip224_ruler_fixed_corr_main"),
          ("Thyroid caliper · DINOv2", "synthetic/thyroid/dino518_caliper_corr_main"),
          ("Thyroid caliper · MedSigLIP", "synthetic/thyroid/medsiglip448_caliper_corr_main"),
          ("Ovary caliper · DINOv2", "synthetic/ovary/dino518_caliper_corr_main"),
          ("Capsule debris · DINOv2", "synthetic/capsule/dino518_debris_corr_main"),
          ("Capsule debris · MedSigLIP", "synthetic/capsule/medsiglip448_debris_corr_main"),
          ("Chest tube · RAD-DINO", "synthetic/nih_ptx/raddino518_tube_corr_main")]


def boot_table(rel):
    f = R / rel / "bootstrap_vs_erm.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f)
    col = "method_a" if "method_a" in b else "arm"
    if "env" in b:
        b = b[b.env == "test_rev"]
    return b.rename(columns={col: "arm"})


def dose_response():
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharey=True)
    for ax, (title, rel) in zip(axes.ravel(), SWEEPS):
        b = boot_table(rel)
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(title, fontsize=8)
        if b is None:
            continue
        for arm, lab, st in (("mask", "ROI masking", "-o"), ("umte", "U-MtE", "--s"), ("umte_balanced", "U-MtE + balanced", ":^")):
            q = b[b.arm == arm].sort_values("overlap")
            if len(q):
                ax.plot(q.overlap, q.seed_delta_mean, st, ms=3, label=lab)
                ax.fill_between(q.overlap, q.ci95_lo, q.ci95_hi, alpha=.15)
        ax.set_xticks([0, .5, 1])
    for ax in axes[1]:
        ax.set_xlabel("artifact–ROI overlap r")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ reversed AUROC vs ERM")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "dose_response.pdf"); fig.savefig(F / "dose_response.png", dpi=200)


def forest():
    d = pd.read_csv(R / "PRIMARY_CLAIMS.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    y = np.arange(len(d))[::-1]
    ok = (d.p_holm < 0.05) & (d.estimate > 0)
    ax.errorbar(d.estimate, y, xerr=[d.estimate - d.ci95_lo, d.ci95_hi - d.estimate], fmt="none", ecolor="grey", lw=1)
    ax.scatter(d.estimate, y, c=np.where(ok, "tab:blue", "tab:red"), zorder=3, s=18)
    ax.axvline(0, color="k", lw=.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{f.split(' ', 1)[1]} — {c}" for f, c in zip(d.family, d.cohort)], fontsize=7)
    ax.set_xlabel("Δ reversed-test AUROC (95 % CI)\nblue: supported after Holm correction; red: not supported", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "forest.pdf"); fig.savefig(F / "forest.png", dpi=200)


def examples():
    """Openly licensed cohorts only (MMOTU, SEE-AI: CC BY 4.0): in-ROI vs out-of-ROI artifacts with ROI contour."""
    import cv2
    tiles = []
    for csv, cdir, art, col in ((paths.DATA / "ovary/ovary_cohort.csv", paths.DATA / "ovary/cache_518", "marker.npy", "marker_px"),
                                (paths.DATA / "capsule/capsule_cohort.csv", paths.DATA / "capsule/cache_518", "contam.npy", "contam_frac")):
        c = pd.read_csv(csv); ids = (cdir / "ids.txt").read_text().split(); idx = {k: j for j, k in enumerate(ids)}
        X = np.load(cdir / ("gray.npy" if (cdir / "gray.npy").exists() else "rgb.npy"), mmap_mode="r")
        roi = np.load(cdir / "roi.npy", mmap_mode="r"); A = np.load(cdir / art, mmap_mode="r")
        thr = 15 if col == "marker_px" else 0.10
        cov_ok = (c.roi_cover < 0.02) if "roi_cover" in c else True
        for sel in (c[(c[col] >= thr) & (c.r >= 0.6)], c[(c[col] >= thr) & (c.r < 0.05) & cov_ok]):
            i = sel.sample(1, random_state=3).image_id.iloc[0]; j = idx[i]
            im = np.asarray(X[j]); im = np.repeat(im[..., None], 3, -1) if im.ndim == 2 else im.copy()
            cnts, _ = cv2.findContours(np.asarray(roi[j]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cnts, -1, (0, 255, 0), 2)
            m = np.asarray(A[j]) > 0; im[m] = (0.4 * im[m] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            tiles.append(im)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, t, lab in zip(axes, tiles, ["Ovary: caliper in ROI", "Ovary: overlay outside ROI", "Capsule: debris on lesion",
                                        "Capsule: debris off lesion"]):
        ax.imshow(t); ax.set_title(lab, fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(F / "examples.pdf"); fig.savefig(F / "examples.png", dpi=200)


if __name__ == "__main__":
    dose_response(); forest(); examples(); print("figures written to", F)
