"""I2E controls on the synthetic cohort: is the gain shortcut removal or generic regularisation?

For a given backbone: apply (none | I2E | random subspace of equal rank | top-k PCA) and fit
  (a) a head on *clean* training data (no shortcut)  -> clean AUROC
  (b) the trap head at a given overlap               -> clean and reversed AUROC, gap closure
"""
import argparse

import numpy as np
import pandas as pd

from wtss import paths
from wtss.data.isic2018 import load_isic2018_pilot
from wtss.features import assemble_env
from wtss.heads import fit_on_transformed
from wtss.methods.insertion import SubspaceEraser, fit_difference_subspace
from wtss.stats import safe_auc
from wtss.synthetic import presence_vector

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="isic2018")
ap.add_argument("--backbone_dir", default="dinov2_b14_518")
ap.add_argument("--artifact", default="ruler_fixed")
ap.add_argument("--insert_view", default="insert_random_ruler")
ap.add_argument("--overlaps", nargs="+", type=float, default=[0.0, 0.5, 1.0])
a = ap.parse_args()

if a.cohort == "isic2018":
    coh = load_isic2018_pilot()
else:
    from wtss.data.cxr import load_nih_synthetic_cohort
    coh = load_nih_synthetic_cohort(518)[0]
F = paths.CACHE / "features" / coh.name / a.backbone_dir / a.artifact
L = lambda n: np.load(F / f"{n}.npz")["X"]
df = coh.df; y = df.y.to_numpy(); ids = df.image_id.to_numpy()
tr, va, te = (np.flatnonzero(df.split == s) for s in ("train", "val", "test"))
Xc, Xi = L("erm_clean"), L(a.insert_view)
er = fit_difference_subspace(Xc[tr], Xi[tr], energy=0.9, seed=0)
rng = np.random.default_rng(0)
mu = Xc[tr].mean(0)
rand = SubspaceEraser(np.linalg.qr(rng.normal(size=(Xc.shape[1], er.k)))[0].astype(np.float32), mu, er.k, np.zeros(1))
pca = SubspaceEraser(np.linalg.svd(Xc[tr] - mu, full_matrices=False)[2][:er.k].T.astype(np.float32), mu, er.k, np.zeros(1))
rows = []
for name, T in [("none", lambda X: X), ("i2e", er), (f"random{er.k}", rand), (f"pca{er.k}", pca)]:
    clf, C, _ = fit_on_transformed(T, Xc[tr], y[tr], Xc[va], y[va], 0)
    clean_trained = safe_auc(y[te], clf.predict_proba(Xc[te])[:, 1])
    for ov in a.overlaps:
        Xa = L(f"erm_ov{int(round(ov * 100)):03d}")
        res = []
        for seed in (42, 123, 456):
            p = {e: presence_vector(ids, y, seed, e) for e in ("train_corr", "test_rev")}
            clf2, _, _ = fit_on_transformed(T, assemble_env(Xc, Xa, p["train_corr"])[tr], y[tr], Xc[va], y[va], seed)
            res.append((safe_auc(y[te], clf2.predict_proba(Xc[te])[:, 1]),
                        safe_auc(y[te], clf2.predict_proba(assemble_env(Xc, Xa, p["test_rev"])[te])[:, 1])))
        c, r = np.mean(res, 0)
        rows.append({"transform": name, "k": er.k, "overlap": ov, "clean_trained_clean_auc": clean_trained,
                     "trap_clean_auc": c, "trap_rev_auc": r, "trap_gap_clean_minus_rev": c - r})
out = paths.ensure(paths.RESULTS / "analysis")
d = pd.DataFrame(rows)
d.to_csv(out / f"i2e_controls_{a.cohort}_{a.backbone_dir}_{a.artifact}.csv", index=False)
print("energy remaining k=1,4,16,32,64:", er.energy_curve[[1, 4, 16, 32, 64]].round(3))
print(d.round(3).to_string())
