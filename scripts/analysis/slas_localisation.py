"""SLAS step 1 — can a patch-token probe trained ONLY on synthetic generic overlays find REAL artifacts?

Probe training: hair-free ISIC 2019 images (hair_px_native <= 30) + wtss.synthetic.draw_generic_artifact; the
inserted-pixel mask gives patch labels. Evaluation: disjoint images (different leakage groups) with Wegley
hair/ruler masks; patch positive = Wegley coverage >= 0.2, negative = coverage 0.
Reported: patch AUROC (all patches / ROI patches only), IoU and recall at tau=0.5, for
  generic   probe from synthetic overlays (the method; no real artifact mask used)
  oracle    probe trained on real Wegley masks of other images (ceiling, not a method)

  python scripts/analysis/slas_localisation.py --n_train 2000 --n_eval 1000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from wtss import paths
from wtss.backbones import load_dino
from wtss.experiments.real_traps import RealCache
from wtss.methods.token_suppression import fit_probe, pooled_views
from wtss.synthetic import draw_generic_artifact

C = paths.DATA / "isic2019" / "prepared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("cuda")
    c = pd.read_csv(C / "cohort_spec.csv")
    cache = RealCache(C / "cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
    rng = np.random.default_rng(2026)
    groups = c.group.unique()
    rng.shuffle(groups)
    g_tr = set(groups[: len(groups) // 2])
    tr = c[c.group.isin(g_tr) & (c.hair_px_native <= 30)].image_id.to_numpy()
    tr = rng.choice(tr, min(a.n_train, len(tr)), replace=False)
    ev_pool = c[~c.group.isin(g_tr) & (c.hair_frac >= 0.02) & (c.has_hair_mask == True)]
    ev = rng.choice(ev_pool.image_id.to_numpy(), min(a.n_eval, len(ev_pool)), replace=False)
    orc_pool = c[c.group.isin(g_tr) & (c.hair_frac >= 0.02) & (c.has_hair_mask == True)].image_id.to_numpy()
    orc = rng.choice(orc_pool, min(a.n_train, len(orc_pool)), replace=False)

    def ins(i):
        rgb, roi, _ = cache.get(i)
        out = np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"slas|{i}"))
        m = (np.abs(out.astype(np.int16) - rgb.astype(np.int16)).max(-1) > 8).astype(np.uint8)
        return Image.fromarray(out), m

    def real(i):
        rgb, _, art = cache.get(i)
        return Image.fromarray(rgb), (art > 0).astype(np.uint8)

    def plain(i):
        return Image.fromarray(cache.get(i)[0])

    bk = load_dino(518, dev)
    probes = {"generic": fit_probe(bk.model, dev, tr, ins, bk.preprocess, n_tokens=120_000),
              "oracle": fit_probe(bk.model, dev, orc, real, bk.preprocess, n_tokens=120_000)}
    res = {}
    for name, pr in probes.items():
        v = pooled_views(bk.model, dev, ev, plain, bk.preprocess, lambda i: cache.get(i)[1] > 0, pr,
                         art_fn=lambda i: (cache.get(i)[2] > 0).astype(np.uint8))
        s, cov, roi = v["loc_score"].astype(np.float32), v["loc_cover"].astype(np.float32), v["loc_roi"]
        lab = np.where(cov >= 0.2, 1, np.where(cov == 0, 0, -1))
        r = {}
        for scope, sel in (("all", lab >= 0), ("roi", (lab >= 0) & roi)):
            y, p = lab[sel], s[sel]
            pred = p >= 0.5
            r[scope] = {"auroc": float(roc_auc_score(y, p)), "iou@0.5": float((pred & (y == 1)).sum() / ((pred | (y == 1)).sum())),
                        "recall@0.5": float(pred[y == 1].mean()), "fpr@0.5": float(pred[y == 0].mean()),
                        "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
        res[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    out = paths.ensure(paths.RESULTS / "slas")
    (out / "localisation_isic_hair.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
