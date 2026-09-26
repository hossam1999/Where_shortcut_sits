"""E12 — overlap-contrast trap (the failed design): A = 1 hair with r >= 0.5, A = 0 hair with r < 0.1
(no hair-free group). Spec expectation: null everywhere (mask − ERM +0.047; control +0.019 [−0.085, +0.122])."""
import json

import torch

from wtss import paths
from wtss.data.isic2019_spec import build_spec_envs, load_spec_cohort
from wtss.experiments.real_traps import RealCache
from wtss.experiments.spec_traps import run_spec
from wtss.stats import hierarchical_paired_bootstrap
import pandas as pd

c = load_spec_cohort()
c["contrast_A0"] = ~c.A0 & (c.r_spec < 0.1)
c["contrast_A1"] = ~c.A0 & (c.r_spec >= 0.5)
envs = build_spec_envs(c, traps=("contrast",))
out = paths.ensure(paths.RESULTS / "spec_e13" / "dino518_e12_contrast")
if not (out / "predictions.csv.gz").exists():
    cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", roi_file="roi_spec.npy")
    run_spec(envs, cache, "dino518", out, paths.CACHE / "features" / "spec_isic2019" / "dinov2_b14_518_e12", [],
             arms=("erm", "mask", "balanced", "dfr"), traps=("contrast",), device=torch.device("cuda"))
p = pd.read_csv(out / "predictions.csv.gz")
res = {a: {k: v for k, v in hierarchical_paired_bootstrap(p, a, "erm", "test_rev", 10000, 5, fast=True).items()
           if k in ("seed_delta_mean", "ci95_lo", "ci95_hi")} for a in ("mask", "balanced", "dfr")}
(out / "E12.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=1, default=float))
