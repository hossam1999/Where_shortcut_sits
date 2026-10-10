"""Draws the two explanatory diagrams of status report 1 (no data needed).

  python status_reports/report1_location_of_artifacts/make_figures.py
    -> figures/masking_idea.pdf, figures/two_traps.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
SKIN, LESION, ROI, ART, GREY = "#f2d4c4", "#8a5a44", "#00a000", "#dc0000", "#808080"


def panel(ax, title, ruler_xy, masked):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 8, color=GREY if masked else SKIN))
    if masked:
        ax.add_patch(Ellipse((5, 4), 5.2, 4.0, color=SKIN))
    ax.add_patch(Ellipse((5, 4), 4.6, 3.4, color=LESION))
    ax.add_patch(Ellipse((5, 4), 5.2, 4.0, fill=False, ec=ROI, lw=2.5))
    x, y = ruler_xy
    inside = 2.6 < x < 6.4 and 2.4 < y < 5.4
    if not masked or inside:
        ax.add_patch(Rectangle((x, y), 1.6, 0.45, color="white", ec=ART, lw=2))
    ax.set_title(title, fontsize=10)


def masking_idea():
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    panel(axs[0, 0], "Ruler outside the lesion", (7.6, 0.8), False)
    panel(axs[0, 1], "After masking: ruler removed", (7.6, 0.8), True)
    panel(axs[1, 0], "Ruler on the lesion", (4.2, 3.8), False)
    panel(axs[1, 1], "After masking: ruler kept,\neverything else removed", (4.2, 3.8), True)
    for r in range(2):
        fig.add_artist(FancyArrowPatch((0.485, 0.75 - 0.47 * r), (0.535, 0.75 - 0.47 * r), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=18, color="black"))
    fig.text(0.5, 0.01, "green line = region of interest (ROI); red box = artifact; grey = removed by the mask",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "masking_idea.pdf", bbox_inches="tight")


def box(ax, x, y, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="black", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def two_traps():
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 4.9, 2.0, 3.8, 1.2, "Images with NO artifact\n(the same images in both traps)", "#e8e8e8", 8.5)
    box(ax, 0.1, 3.7, 4.0, 1.2, "Images with the artifact\nINSIDE the ROI ($r \\geq 0.5$)", "#f6c9c9", 8.5)
    box(ax, 0.1, 0.3, 4.0, 1.2, "Images with the artifact\nOUTSIDE the ROI ($r < 0.1$)", "#cfe8cf", 8.5)
    box(ax, 10.3, 3.7, 3.5, 1.2, "Trap A\n(artifact inside)", "#f6c9c9", 10)
    box(ax, 10.3, 0.3, 3.5, 1.2, "Trap B\n(artifact outside)", "#cfe8cf", 10)
    arrows = [((8.7, 3.0), (10.3, 3.9)), ((8.7, 2.2), (10.3, 1.3)), ((4.1, 4.3), (10.3, 4.3)), ((4.1, 0.9), (10.3, 0.9))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color="black", lw=1))
    fig.text(0.5, 0.0, "$r$ = share of the artifact's pixels that lie inside the ROI; images with $0.1 \\leq r < 0.5$ "
             "are used by neither trap", ha="center", fontsize=8.5, style="italic")
    fig.savefig(OUT / "two_traps.pdf", bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    masking_idea()
    two_traps()
