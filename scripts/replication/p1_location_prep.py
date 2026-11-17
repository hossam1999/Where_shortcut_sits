"""P1 location test (amendment A15): lungs + heart ROI and tube placements for P1's No-SPC seed-1234 split.

Input:  $WTSS_DATA/replication/viviano/location/nospc_1234.csv (their JointDataset, ratio 0.5, seed 1234; listed in P1 env)
Output: .../location/placements_1234.csv: per image the inside (overlap with lungs u heart >= 0.95) and outside
        (overlap <= 0.02 with lungs u heart u P1's circle) tube boxes at 224 px (43 x 6, TUBE_GEOMETRY[224]), achieved
        overlaps, and .../location/roi224_1234.npz (ids, lungs u heart masks at 224 px) for our arms.
PSPNet (TorchXRayVision chestx_det) at 512 px from the 224-px source, p > 0.5; GPU memory capped (shared GPU).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wtss.data.cxr import TUBE_GEOMETRY  # noqa: E402
from wtss.synthetic import find_best_placements_for_mask  # noqa: E402

D = os.environ.get("WTSS_DATA", "/workspace/data")
L = Path(f"{D}/replication/viviano/location")
S = 224


def circle(size=S):
    """P1's mask: skimage.draw.ellipse(c, c, size/2.5, size/2.5) (their xray.py), drawn here at 224 px."""
    yy, xx = np.mgrid[:size, :size]
    c = size // 2
    return (((yy - c) / (size / 2.5)) ** 2 + ((xx - c) / (size / 2.5)) ** 2 < 1).astype(np.uint8)


def main():
    torch.cuda.set_per_process_memory_fraction(0.15)
    import torchxrayvision as xrv
    df = pd.read_csv(L / "nospc_1234.csv")
    keys = (df.site.astype(str) + "/" + df.image_id).tolist()
    m = xrv.baseline_models.chestx_det.PSPNet().eval().cuda()
    keep = [m.targets.index(t) for t in ("Left Lung", "Right Lung", "Heart")]
    roi = np.zeros((len(df), S, S), np.uint8)
    with torch.inference_mode():
        for k in range(0, len(df), 8):
            g = []
            for p in df.path[k:k + 8]:
                a = cv2.imread(f"{D}/cxr_at/{p}", cv2.IMREAD_UNCHANGED)
                a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if a.dtype != np.uint8 else a
                a = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                g.append(cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA))
            x = torch.from_numpy(np.stack(g).astype(np.float32) / 255 * 2048 - 1024)[:, None].cuda()
            r = (torch.sigmoid(m(x))[:, keep].amax(1) > 0.5).cpu().numpy().astype(np.uint8)
            roi[k:k + 8] = np.stack([cv2.resize(z, (S, S), interpolation=cv2.INTER_NEAREST) for z in r])
            if k % 1600 == 0:
                print(f"{k}/{len(df)}", flush=True)
    np.savez_compressed(L / "roi224_1234.npz", ids=np.asarray(keys, dtype="U"), masks=roi)
    w, h = TUBE_GEOMETRY[S]
    circ = circle()
    rows = []
    for key, r in zip(keys, roi):
        pin = find_best_placements_for_mask(r, key, [1.0], w, h)["1.00"]
        pout = find_best_placements_for_mask(np.maximum(r, circ), key, [0.0], w, h)["0.00"]
        rows.append({"key": key, "in_x": pin["x"], "in_y": pin["y"], "in_overlap": pin["achieved"],
                     "out_x": pout["x"], "out_y": pout["y"], "out_overlap_roi_or_circle": pout["achieved"]})
    p = pd.DataFrame(rows)
    p["in_feasible"] = p.in_overlap >= 0.95
    p["out_feasible"] = p.out_overlap_roi_or_circle <= 0.02
    p.to_csv(L / "placements_1234.csv", index=False)
    print(f"ROI mean fraction {roi.mean():.3f}; inside feasible {p.in_feasible.mean():.4f} "
          f"({(~p.in_feasible).sum()} not); outside feasible {p.out_feasible.mean():.4f} ({(~p.out_feasible).sum()} not)")


if __name__ == "__main__":
    main()
