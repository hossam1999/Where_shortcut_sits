"""Diagnostics for real-artifact traps (post-hoc, descriptive; not pre-registered claims).

1. Pixel share of the shortcut: the ERM head evaluated on hair-inpainted test images (removal at test only).
2. Hair-pixel sensitivity per arm: mean |p(original) - p(hair-inpainted)| on artifact-present test images,
   same trained head (the thesis's counterfactual-sensitivity measure, now on real hair).
3. Metadata correlates of A (anatomical site, sex, age) within source: is 'hair present' a proxy for
   patient/site attributes that no pixel-level method can remove?
"""
import argparse

import numpy as np
import pandas as pd

from wtss import heads as H
from wtss import paths
from wtss.data.isic2019 import build_trap_envs, load_cohort
from wtss.methods.insertion import fit_difference_subspace
from wtss.stats import safe_auc

ap = argparse.ArgumentParser()
ap.add_argument("--backbone_dir", default="dinov2_b14_518")
a = ap.parse_args()
F = paths.CACHE / "features" / "real_isic2019" / a.backbone_dir
V = {v: np.load(F / f"{v}.npz") for v in ("erm", "inpaint", "insert")}
ids = V["erm"]["ids"].astype(str)
pos = {k: j for j, k in enumerate(ids)}
X = {v: z["X"] for v, z in V.items()}
df = load_cohort()
envs = build_trap_envs(df)


def get(view, d):
    return X[view][[pos[i] for i in d.image_id]]


rows, sens = [], []
for trap in ("trapA", "trapB"):
    for k in range(5):
        E = {e: envs[(trap, k, e)] for e in ("train_corr", "test_rev", "clean_test", "clean_val", "val_groups", "train_all")}
        tr, cv, te, ct = E["train_corr"], E["clean_val"], E["test_rev"], E["clean_test"]
        ytr, atr, yv = tr.y.to_numpy(), tr.a.to_numpy(), cv.y.to_numpy()
        Xtr, Xv = get("erm", tr), get("erm", cv)
        ta = E["train_all"]
        heads = {"erm": H.fit_erm(Xtr, ytr, Xv, yv, k)[0], "balanced": H.fit_balanced(Xtr, ytr, atr, Xv, yv, k)[0]}
        g = E["val_groups"]
        heads["dfr"] = H.fit_dfr(get("erm", g), g.y.to_numpy(), g.a.to_numpy(), Xv, yv, k)[0]
        hp = ta[ta.a == 1]
        heads["leace_paired"] = H.fit_on_transformed(H.eraser_fn(H.fit_leace(get("inpaint", hp), get("erm", hp))), Xtr, ytr, Xv, yv, k)[0]
        er = fit_difference_subspace(get("erm", ta), get("insert", ta), energy=0.9, seed=k)
        heads["i2e"] = H.fit_on_transformed(er, Xtr, ytr, Xv, yv, k)[0]
        # 1. pixel share: ERM head with hair removed at test only
        yt = te.y.to_numpy()
        rows.append({"trap": trap, "fold": k, "erm_rev": safe_auc(yt, heads["erm"].predict_proba(get("erm", te))[:, 1]),
                     "erm_rev_inpainted_at_test": safe_auc(yt, heads["erm"].predict_proba(get("inpaint", te))[:, 1]),
                     "erm_clean": safe_auc(ct.y.to_numpy(), heads["erm"].predict_proba(get("erm", ct))[:, 1]),
                     "i2e_energy_remaining_k64": float(er.energy_curve[-1])})
        # 2. hair-pixel sensitivity
        art = te[te.a == 1]
        for name, h in heads.items():
            dp = np.abs(h.predict_proba(get("erm", art))[:, 1] - h.predict_proba(get("inpaint", art))[:, 1])
            sens.append({"trap": trap, "fold": k, "method": name, "mean_abs_dp_hair": float(dp.mean())})
r = pd.DataFrame(rows)
r["pixel_share_of_gap"] = (r.erm_rev_inpainted_at_test - r.erm_rev) / (r.erm_clean - r.erm_rev)
s = pd.DataFrame(sens).groupby(["trap", "method"]).mean_abs_dp_hair.mean().unstack(0)
# 3. metadata correlates
m = df[df.group_A.isin(["free", "trapA", "trapB"])]  # cohort already carries ISIC 2019 metadata
site = pd.crosstab([m.source, m.group_A], m.anatom_site_general.fillna("NA"), normalize="index").round(2)
sex = pd.crosstab([m.source, m.group_A], m.sex.fillna("NA"), normalize="index").round(2)
age = m.groupby(["source", "group_A"]).age_approx.mean().round(1)
out = paths.ensure(paths.RESULTS / "analysis")
r.to_csv(out / f"trap_pixel_share_{a.backbone_dir}.csv", index=False)
s.to_csv(out / f"trap_hair_sensitivity_{a.backbone_dir}.csv")
site.to_csv(out / "trap_metadata_site.csv"); sex.to_csv(out / "trap_metadata_sex.csv"); age.to_csv(out / "trap_metadata_age.csv")
print(r.groupby("trap").mean(numeric_only=True).round(3).T)
print("\nhair-pixel sensitivity |Δp| (same head, original vs hair-inpainted):\n", s.round(3))
print("\nsite mix:\n", site.to_string()); print("\nsex:\n", sex.to_string()); print("\nage:\n", age.to_string())
