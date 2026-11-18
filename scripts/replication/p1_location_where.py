"""Where the P1 location-test markers sit at the outside position (A15), per split: above the lungs + heart ROI, below it,
or beside it (left/right), from the saved placements and PSPNet ROIs (224 px). Description only; no analysis depends on it.

    python scripts/replication/p1_location_where.py
Writes results/replication/viviano/location_outside_regions.csv.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
L = Path(f"{D}/replication/viviano/location")
ROOT = Path(__file__).resolve().parents[2]
H, SIDE = 6, 224   # marker height and image side at 224 px (TUBE_GEOMETRY)

p = pd.read_csv(L / "placements_1234.csv")
s = pd.read_csv(L / "nospc_1234.csv")
s["key"] = s.site.astype(str) + "/" + s.image_id
p = p.merge(s[["key", "split"]], on="key")
z = np.load(L / "roi224_1234.npz")
roi = dict(zip(z["ids"], z["masks"]))


def region(r):
    rows = np.where(roi[r.key].any(1))[0]
    if not len(rows):
        return "empty ROI"
    if r.out_y + H <= rows.min():
        return "above ROI"
    return "below ROI" if r.out_y > rows.max() else "beside ROI"


p["region"] = p.apply(region, axis=1)
p["px_to_top_edge"] = p.out_y
p["px_to_bottom_edge"] = SIDE - p.out_y - H
out = []
for sp, g in p.groupby("split"):
    for reg, gg in g.groupby("region"):
        out.append({"split": sp, "region": reg, "n": len(gg), "share": round(len(gg) / len(g), 4),
                    "median_px_to_nearest_top_or_bottom_edge": float(np.minimum(gg.px_to_top_edge, gg.px_to_bottom_edge).median())})
t = pd.DataFrame(out)
t.to_csv(ROOT / "results/replication/viviano/location_outside_regions.csv", index=False)
print(t.to_string(index=False))
