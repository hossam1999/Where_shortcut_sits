"""Design schematic of the paper (Fig. design): controlled sweep, real traps, real-artifact transplant, environments,
arms and the location crossover. Drawn from primitives; no data.
  python scripts/make_schematic.py      -> paper/figures/design.pdf (+ .png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
ROI, ART, MASK, ERM, INK = "#00a000", "#dc0000", "#1f77b4", "#6e6e6e", "#222222"
SKIN, LESION, NEUTRAL = "#f3e6d8", "#b88a6a", "#9a9a9a"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})


def image(ax, x, y, s=1.0, masked=False, lesion=True):
    """One schematic image at (x, y) of side s: skin background, lesion (ROI) ellipse; masked = background removed."""
    ax.add_patch(Rectangle((x, y), s, s, fc="#d9d9d9" if masked else SKIN, ec=INK, lw=0.6,
                           hatch="////" if masked else None))
    if lesion:
        ax.add_patch(Ellipse((x + 0.5 * s, y + 0.5 * s), 0.52 * s, 0.42 * s, fc=LESION, ec=ROI, lw=1.2))


def ruler(ax, x, y, s, cx, cy, colour=ART, w=0.34, h=0.07):
    ax.add_patch(Rectangle((x + (cx - w / 2) * s, y + (cy - h / 2) * s), w * s, h * s, fc=colour, ec="none"))


def strand(ax, x, y, s, pts, colour=ART, lw=1.4):
    verts = [(x + a * s, y + b * s) for a, b in pts]
    codes = [MPath.MOVETO] + [MPath.CURVE3] * (len(verts) - 1)
    ax.add_patch(PathPatch(MPath(verts, codes), fc="none", ec=colour, lw=lw, capstyle="round"))


IN_STRAND = [(0.34, 0.42), (0.50, 0.66), (0.64, 0.50)]    # on the lesion
OUT_STRAND = [(0.08, 0.14), (0.20, 0.30), (0.28, 0.10)]   # in the background


def panel_title(ax, x, y, text):
    ax.text(x, y, text, fontsize=8, fontweight="bold", ha="left", va="bottom", color=INK)


def main():
    fig = plt.figure(figsize=(7.2, 2.95))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 26.4); ax.set_ylim(0, 10.6); ax.axis("off")
    s = 1.8

    # (a) controlled sweep
    panel_title(ax, 0.2, 9.75, "(a) Controlled sweep")
    ax.text(0.2, 9.25, "synthetic artifact, controlled overlap $r$", fontsize=6.3, color=ERM)
    for k, (r, cx, cy) in enumerate(((0.0, 0.5, 0.12), (0.5, 0.5, 0.29), (1.0, 0.5, 0.5))):
        x0 = 0.2 + k * 2.05
        image(ax, x0, 6.8, s)
        ruler(ax, x0, 6.8, s, cx, cy)
        ax.text(x0 + s / 2, 6.5, f"$r={r:g}$", ha="center", va="top")
    ax.annotate("", xy=(6.1, 5.75), xytext=(0.3, 5.75), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax.text(3.2, 5.45, "artifact moves into the ROI", ha="center", va="top", fontsize=6.3)

    # (b) real traps
    panel_title(ax, 6.9, 9.75, "(b) Real traps")
    ax.text(6.9, 9.25, "different images carry the artifact", fontsize=6.3, color=ERM)
    image(ax, 7.1, 6.8, s); strand(ax, 7.1, 6.8, s, IN_STRAND)
    image(ax, 9.6, 6.8, s); strand(ax, 9.6, 6.8, s, OUT_STRAND)
    ax.text(8.0, 6.5, "Trap A\n$r\\geq0.5$", ha="center", va="top")
    ax.text(10.5, 6.5, "Trap B\n$r<0.1$", ha="center", va="top")
    image(ax, 8.55, 3.2, 1.5)
    ax.text(9.3, 2.9, "shared artifact-free\ngroup", ha="center", va="top", fontsize=6.3)
    for xe in (8.0, 10.5):
        ax.plot([xe, 9.3], [5.55, 4.8], color=ERM, lw=0.6)

    # (c) transplant
    panel_title(ax, 12.6, 9.75, "(c) Transplant")
    ax.text(12.6, 9.25, "one real artifact, same image, two places", fontsize=6.3, color=ERM)
    image(ax, 12.8, 6.8, s); strand(ax, 12.8, 6.8, s, IN_STRAND)
    image(ax, 15.3, 6.8, s); strand(ax, 15.3, 6.8, s, OUT_STRAND)
    ax.text(13.7, 6.5, "inside\n$r\\geq0.95$", ha="center", va="top")
    ax.text(16.2, 6.5, "outside\n$r=0$", ha="center", va="top")
    image(ax, 14.2, 3.2, 1.5); strand(ax, 14.2, 3.2, 1.5, IN_STRAND, colour=NEUTRAL, lw=2.2)
    ax.text(14.95, 2.9, "control: neutral tissue\nthrough the same mask", ha="center", va="top", fontsize=6.3)

    # (d) arms and environments
    panel_title(ax, 18.6, 9.75, "(d) Arms and environments")
    image(ax, 18.9, 7.2, 1.5); strand(ax, 18.9, 7.2, 1.5, OUT_STRAND)
    image(ax, 22.4, 7.2, 1.5, masked=True)
    ax.add_patch(Ellipse((22.4 + 0.75, 7.2 + 0.75), 0.78, 0.63, fc=LESION, ec=ROI, lw=1.2))
    ax.text(19.65, 6.95, "ERM\n(whole image)", ha="center", va="top", fontsize=6.3, color=ERM)
    ax.text(23.15, 6.95, "ROI masking\n(outside removed)", ha="center", va="top", fontsize=6.3, color=MASK)
    ax.text(18.8, 5.45, "artifact rate in $Y{=}1$ / $Y{=}0$", fontsize=6.3, va="bottom", color=ERM)
    rows = [("train", "0.9 / 0.1"), ("correlated test", "0.9 / 0.1"), ("reversed test", "0.1 / 0.9"), ("clean test", "independent")]
    for k, (lab, v) in enumerate(rows):
        yk = 4.95 - 0.52 * k
        bold = "bold" if lab == "reversed test" else "normal"
        ax.text(18.9, yk, lab, fontsize=6.5, va="center", fontweight=bold)
        ax.text(25.9, yk, v, fontsize=6.5, va="center", ha="right", fontweight=bold)

    # crossover definition across the bottom of (b)-(d)
    ax.add_patch(FancyBboxPatch((6.95, 0.2), 19.0, 1.55, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc="#f4f8fc", ec=MASK, lw=0.7))
    ax.text(16.45, 1.3, "location crossover $=\\;[\\mathrm{mask}-\\mathrm{ERM}]_{\\mathrm{outside}}\\;-\\;"
            "[\\mathrm{mask}-\\mathrm{ERM}]_{\\mathrm{inside}}$  (reversed-test AUROC)", ha="center", va="center", fontsize=6.8)
    ax.text(16.45, 0.62, "positive when masking removes a shortcut outside the ROI but not one inside it",
            ha="center", va="center", fontsize=6.3, color=ERM)
    # legend under (a)
    ax.add_patch(Ellipse((0.75, 3.95), 0.7, 0.45, fc=LESION, ec=ROI, lw=1.2))
    ax.text(1.3, 3.95, "ROI (lesion, nodule, tumour, lung)", va="center", fontsize=6.1)
    ax.add_patch(Rectangle((0.4, 2.87), 0.7, 0.16, fc=ART, ec="none"))
    ax.text(1.3, 2.95, "artifact (ruler, hair, caliper, debris)", va="center", fontsize=6.1)
    ax.add_patch(Rectangle((0.4, 1.7), 0.7, 0.45, fc="#d9d9d9", ec=INK, lw=0.5, hatch="////"))
    ax.text(1.3, 1.92, "removed by masking", va="center", fontsize=6.1)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "design.pdf"); fig.savefig(OUT / "design.png", dpi=220)
    print("wrote", OUT / "design.pdf")


if __name__ == "__main__":
    main()
