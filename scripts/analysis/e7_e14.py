"""E7 (lesion-size tertiles, synthetic ruler) and E14 (artifact-to-lesion area ratio, real hair Trap A).

Descriptive, from saved predictions: within each stratum, mask − ERM reversed AUROC per seed (image IDs
identical across arms), mean over seeds; E14 also with the hierarchical bootstrap.
"""
import json

import numpy as np
import pandas as pd

from wtss import paths
from wtss.data.isic2018 import load_isic2018_pilot
from wtss.stats import hierarchical_paired_bootstrap, safe_auc
from wtss.synthetic import ARTIFACT_GEOMETRY

out = paths.ensure(paths.RESULTS / "analysis")

# ---------------- E7
coh = load_isic2018_pilot().df
roi = np.load(paths.CACHE / "images" / "isic2018_pilot_518" / "roi.npy", mmap_mode="r")
ids = (paths.CACHE / "images" / "isic2018_pilot_518" / "ids.txt").read_text().split()
area = pd.Series([float(np.asarray(roi[j]).mean()) for j in range(len(ids))], index=ids)
p = pd.read_csv(paths.RESULTS / "synthetic" / "isic2018" / "dino518_ruler_fixed_corr_main" / "predictions.csv.gz")
test_ids = coh[coh.split == "test"].image_id
tert = pd.qcut(area[test_ids], 3, labels=["small", "medium", "large"])
w, h = ARTIFACT_GEOMETRY[518]
rows = []
for ov in (0.0, 0.5, 1.0):
    for t in ("large", "medium", "small"):
        keep = set(tert[tert == t].index)
        ds = []
        for s in (42, 123, 456):
            q = p[(p.env == "test_rev") & (p.seed == s) & np.isclose(p.overlap, ov) & p.image_id.isin(keep)]
            e, m = q[q.method == "erm"].set_index("image_id"), q[q.method == "mask"].set_index("image_id")
            ds.append(safe_auc(m.y, m.prob) - safe_auc(e.y, e.prob))
        rows.append({"overlap": ov, "tertile": t, "n_mel": int(q[q.method == "erm"].y.sum()),
                     "mask_minus_erm": float(np.mean(ds)),
                     "artifact_to_lesion_ratio_median": float(np.median(w * h / (area[list(keep)] * 518 * 518)))})
e7 = pd.DataFrame(rows)
e7.to_csv(out / "E7_lesion_tertiles.csv", index=False)
print("E7\n", e7.pivot(index="overlap", columns="tertile", values="mask_minus_erm")[["large", "medium", "small"]].round(3))
print(e7.groupby("tertile").artifact_to_lesion_ratio_median.first().round(3))

# ---------------- E14
c = pd.read_csv(paths.DATA / "isic2019" / "prepared" / "cohort_spec.csv")
c["ratio"] = c.hair_frac_256 / c.lesion_frac_spec.replace(0, np.nan)
sp = pd.read_csv(paths.RESULTS / "spec_e13" / "dino518_spec" / "predictions.csv.gz")
sp = sp[sp.trap == "trapA"].merge(c[["image_id", "ratio"]], on="image_id", how="left")
a1 = c[(c.hair_px_native > 30) & (c.r_spec >= 0.5)]
cuts = a1.ratio.quantile([1 / 3, 2 / 3]).to_numpy()
sp["band"] = np.where(sp.artifact_present == 0, "A0", np.where(sp.ratio <= cuts[0], "T1", np.where(sp.ratio <= cuts[1], "T2", "T3")))
res = {"hair_ratio_median_trapA": float(a1.ratio.median()),
       "tertile_medians": [float(a1.ratio[a1.ratio <= cuts[0]].median()), float(a1.ratio[(a1.ratio > cuts[0]) & (a1.ratio <= cuts[1])].median()),
                           float(a1.ratio[a1.ratio > cuts[1]].median())]}
for t in ("T1", "T2", "T3"):
    q = sp[sp.band.isin(["A0", t])]
    b = hierarchical_paired_bootstrap(q, "mask", "erm", "test_rev", 5000, 3, fast=True)
    res[t] = {k: b[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}
(out / "E14_area_ratio.json").write_text(json.dumps(res, indent=2, default=float))
print("E14", json.dumps(res, indent=1, default=float))
