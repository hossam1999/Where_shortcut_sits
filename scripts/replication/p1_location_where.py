"""Where the P1 location-test markers sit at the outside position (A15), per split: above the lungs + heart ROI, below it,
or beside it (left/right), from the saved placements and PSPNet ROIs (224 px). Description only; no analysis depends on it.

    python scripts/replication/p1_location_where.py
Writes results/replication/viviano/location_outside_regions.csv and location_edge_distance.csv (inside vs outside: distance
of the marker box to the nearest image edge and of its centre to the image centre).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
L = Path(f"{D}/replication/viviano/location")
ROOT = Path(__file__).resolve().parents[2]
W, H, SIDE = 43, 6, 224   # marker height and image side at 224 px (TUBE_GEOMETRY)

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

edge = lambda x, y: np.minimum.reduce([x, y, SIDE - x - W, SIDE - y - H])
centre = lambda x, y: np.hypot(x + W / 2 - SIDE / 2, y + H / 2 - SIDE / 2)
e = []
for sp, g in p.groupby("split"):
    for pos, (x, y) in {"inside": (g.in_x, g.in_y), "outside": (g.out_x, g.out_y)}.items():
        d = edge(x, y)
        e.append({"split": sp, "position": pos, "n": len(g), "median_px_to_nearest_edge": float(np.median(d)),
                  "share_within_25px_of_an_edge": round(float((d <= 25).mean()), 4),
                  "median_px_centre_to_image_centre": round(float(np.median(centre(x, y))), 1)})
e = pd.DataFrame(e)
e.to_csv(ROOT / "results/replication/viviano/location_edge_distance.csv", index=False)
print(e.to_string(index=False))
