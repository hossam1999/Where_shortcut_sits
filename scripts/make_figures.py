"""Paper figures, generated from results/ (never edited by hand).

  python scripts/make_figures.py            # -> paper/figures/{dose_response,forest,examples}.pdf/.png
  python scripts/make_figures.py examples   # only the real-image example figure
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


# Example figure (revision 2026-11-23): colour-blind-safe Okabe-Ito colours, 2x2 layout, arrows, and a deterministic
# selection rule instead of a random draw. The earlier random draw picked an ovary "outside" image whose detections
# were probe-arc and screen-edge overlays rather than a caliper; the rule below requires a caliper-like component.
ROI_RGB, ART_RGB = (86, 180, 233), (230, 159, 0)   # Okabe-Ito sky blue (ROI) and orange (artifact)
IMG = 518


def _components(mask, join_px=6):
    """Connected components of a (dotted) artifact mask after joining dots that are <= join_px apart.
    Returns a list of dicts: pixels (bool array), bbox (x0, y0, x1, y1), extent (longest bbox side, px), elong."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join_px + 1, 2 * join_px + 1))
    n, lab = cv2.connectedComponents(cv2.dilate(m, k))
    out = []
    for c in range(1, n):
        px = (lab == c) & (m > 0)
        if not px.any():
            continue
        ys, xs = np.nonzero(px)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        out.append({"pixels": px, "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "extent": int(max(w, h)), "elong": float(max(w, h) / max(1, min(w, h)))})
    return out


def caliper_score(mask, roi, inside, min_extent=60, border=20, top_band=0.12, gap=8):
    """Extent (px) of the longest caliper-like component on the requested side of the ROI, else 0.
    Caliper-like: a dotted line at least min_extent px long and elongated (>= 3:1), clear of the image border and of
    the top band where ultrasound probe arcs and screen text sit. Outside: no pixel within `gap` px of the ROI."""
    import cv2
    r = (np.asarray(roi) > 0).astype(np.uint8)
    near = cv2.dilate(r, np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0
    best = 0
    for c in _components(mask):
        x0, y0, x1, y1 = c["bbox"]
        if c["extent"] < min_extent or c["elong"] < 3:
            continue
        if x0 < border or y0 < max(border, top_band * IMG) or x1 > IMG - border or y1 > IMG - border:
            continue
        frac_in = (c["pixels"] & (r > 0)).sum() / c["pixels"].sum()
        if (inside and frac_in >= 0.6) or (not inside and not (c["pixels"] & near).any()):
            best = max(best, c["extent"])
    return best


def examples():
    """Openly licensed cohorts only (MMOTU, SEE-AI: CC BY 4.0). Thyroid (TN3K, licence not stated) and ISIC (CC BY-NC)
    are excluded from published figures. Images in the blinded visual audit sample are never used.
    Selection (deterministic, recorded in paper/figures/examples_selection.csv):
      ovary in / out: the image with the longest caliper-like component inside the ROI / clear of it (caliper_score);
      capsule on / off lesion: r >= 0.6 / r < 0.05 (and ROI cover < 0.02 where recorded), detector coverage between
      3% and 20% of the image (avoids images dominated by the mask), the one closest to 8% coverage.
    The orange regions are automatic detector outputs, not manual ground truth."""
    import cv2
    audit = paths.REPO_ROOT / "audit" / "sample_ids.csv"
    blind = set(pd.read_csv(audit).image_id.astype(str)) if audit.exists() else set()
    panels, rows = [], []

    def load(csv, cdir, art):
        c = pd.read_csv(csv); c["image_id"] = c.image_id.astype(str); c = c[~c.image_id.isin(blind)]
        ids = (cdir / "ids.txt").read_text().split()
        X = np.load(cdir / ("gray.npy" if (cdir / "gray.npy").exists() else "rgb.npy"), mmap_mode="r")
        return c, {k: j for j, k in enumerate(ids)}, X, np.load(cdir / "roi.npy", mmap_mode="r"), np.load(cdir / art, mmap_mode="r")

    # ovary: score every candidate with an artifact (cheap: 518 px masks)
    c, idx, X, ROI, A = load(paths.DATA / "ovary/ovary_cohort.csv", paths.DATA / "ovary/cache_518", "marker.npy")
    c = c[(c.marker_px >= 15) & c.image_id.isin(idx)]
    for inside, title, label in ((True, "a  Ovary: caliper inside the tumour", "caliper"),
                                 (False, "b  Ovary: caliper outside the tumour", "caliper")):
        cand = c[c.r >= 0.6] if inside else c[c.r < 0.05]
        sc = {i: caliper_score(A[idx[i]], ROI[idx[i]], inside) for i in cand.image_id}
        i = max(sorted(sc), key=lambda k: sc[k])
        if sc[i] == 0:
            raise SystemExit(f"no caliper-like ovary example {'inside' if inside else 'outside'} the ROI; inspect by hand")
        panels.append((idx[i], X, ROI, A, title, label)); rows.append({"panel": title, "image_id": i, "rule_score_px": sc[i]})

    c, idx, X, ROI, A = load(paths.DATA / "capsule/capsule_cohort.csv", paths.DATA / "capsule/cache_518", "contam.npy")
    c = c[c.image_id.isin(idx) & c.contam_frac.between(0.03, 0.20)]
    cov_ok = (c.roi_cover < 0.02) if "roi_cover" in c else True
    for sel, title in ((c[c.r >= 0.6], "c  Capsule: debris on the lesion"), (c[(c.r < 0.05) & cov_ok], "d  Capsule: debris off the lesion")):
        if sel.empty:
            raise SystemExit(f"no capsule candidate for {title}")
        i = sel.assign(d=(sel.contam_frac - 0.08).abs()).sort_values(["d", "image_id"]).image_id.iloc[0]
        panels.append((idx[i], X, ROI, A, title, "debris"))
        rows.append({"panel": title, "image_id": i, "rule_score_px": float(sel.set_index("image_id").contam_frac[i])})

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.6))
    for ax, (j, X, ROI, A, title, label) in zip(axes.ravel(), panels):
        im = np.asarray(X[j]); im = np.repeat(im[..., None], 3, -1) if im.ndim == 2 else im.copy()
        m = np.asarray(A[j]) > 0
        im[m] = (0.35 * im[m] + 0.65 * np.array(ART_RGB)).astype(np.uint8)
        cnts, _ = cv2.findContours(np.asarray(ROI[j]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, ROI_RGB, 3)
        ax.imshow(im); ax.set_title(title, fontsize=10, loc="left"); ax.axis("off")
        comps = sorted(_components(m), key=lambda d: d["pixels"].sum(), reverse=True)
        if comps:
            ys, xs = np.nonzero(comps[0]["pixels"]); tx, ty = float(xs.mean()), float(ys.mean())
            ox = 60 if tx < IMG / 2 else -60; oy = 70 if ty < IMG / 2 else -70
            ax.annotate(label, xy=(tx, ty), xytext=(tx + ox * 1.6, ty + oy), color="white", fontsize=10, fontweight="bold",
                        ha="center", arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
                        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6))
    fig.tight_layout(); fig.savefig(F / "examples.pdf"); fig.savefig(F / "examples.png", dpi=200)
    pd.DataFrame(rows).to_csv(F / "examples_selection.csv", index=False)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["dose_response", "forest", "examples"]   # e.g. `python scripts/make_figures.py examples`
    for name in todo:
        globals()[name]()
    print("figures written to", F)
