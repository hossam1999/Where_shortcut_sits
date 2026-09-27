"""U-MtE design ablation on cached DINOv2 features (docs/PREREGISTRATION_UMTE_ABLATION.md): erasure rank chosen by
held-out energy (0.5 ... 0.99; main = 0.90) or fixed (k = 1 ... 64). Trap A (in-ROI) only; arms erm, mask, mte, mte_protect.
  python scripts/analysis/umte_ablation.py   # -> results/ablation_umte/{cohort}_{setting}/, results/ablation_umte/summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_thyroid_traps import CAP, OV, T, capsule_cohort, cohort, ovary_cohort  # noqa: E402

from wtss import paths  # noqa: E402
from wtss.experiments.real_traps import RealCache  # noqa: E402
from wtss.experiments.spec_traps import run_spec  # noqa: E402
from wtss.data.isic2019_spec import build_spec_envs  # noqa: E402

SETTINGS = [("e0.50", {"mte_energy": 0.50}), ("e0.70", {"mte_energy": 0.70}), ("e0.80", {"mte_energy": 0.80}),
            ("e0.90", {"mte_energy": 0.90}), ("e0.95", {"mte_energy": 0.95}), ("e0.99", {"mte_energy": 0.99}),
            ("k1", {"mte_k": 1}), ("k4", {"mte_k": 4}), ("k16", {"mte_k": 16}), ("k64", {"mte_k": 64})]
COH = {"thyroid": (cohort, T, "marker.npy"), "ovary": (ovary_cohort, OV, "marker.npy"), "capsule": (capsule_cohort, CAP, "contam.npy")}


def main():
    out = paths.ensure(paths.RESULTS / "ablation_umte")
    rows = []
    for co, (fn, d, art) in COH.items():
        c = fn()
        envs = build_spec_envs(c, group_col="group")
        cache = RealCache(d / "cache_518", roi_file="roi.npy", art_file=art)
        for name, ctx in SETTINGS:
            o = out / f"{co}_{name}"
            if not (o / "metrics_per_seed.csv").exists():
                run_spec(envs, cache, "dino518", o, paths.CACHE / "features" / co / "dinov2_b14_518", [],
                         arms=("erm", "mask", "mte", "mte_protect"), traps=("trapA",), device=torch.device("cuda"),
                         insert_fn=lambda *a: None, insert_tag="_generic", extra_ctx=ctx)
            m = pd.read_csv(o / "metrics_per_seed.csv").groupby(["method", "env"]).auc.mean().unstack()
            for arm in ("mask", "mte", "mte_protect"):
                rows.append({"cohort": co, "setting": name, "arm": arm, **m.loc[arm].to_dict()})
            print(co, name, m.round(3).loc[["mask", "mte", "mte_protect"]].to_dict("index"), flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
