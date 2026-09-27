"""Run a synthetic-overlap experiment and its bootstrap analysis.

Examples
  # thesis Result 1 + unified mitigation table (DINOv2 @518)
  python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag main \
      --arms erm mask inpaint balanced groupdro dfr leace inpaint_consistency inpaint_consistency_lam0
  # occlusion control (50/50), dilation, appearance stress, other backbones: see Makefile
"""
from __future__ import annotations

import argparse
import json

import torch

from wtss import paths
from wtss.experiments.synthetic import SynthConfig, analyse_synthetic, run_synthetic
from wtss.synthetic import ARTIFACT_GEOMETRY


def cohort_and_placements(name: str, size: int):
    if name == "isic2018":
        from wtss.data.isic2018 import isic2018_loaders, load_isic2018_pilot

        w, h = ARTIFACT_GEOMETRY[size]
        pl = json.loads((paths.FROZEN / f"placements_{size}_{w}x{h}.json").read_text())
        return load_isic2018_pilot(), pl, isic2018_loaders(size), None
    if name == "nih_ptx":
        from wtss.data.cxr import load_nih_synthetic_cohort

        return load_nih_synthetic_cohort(size)
    if name == "thyroid":
        from wtss.data.thyroid import load_thyroid_synthetic_cohort

        return load_thyroid_synthetic_cohort(size)
    if name == "capsule":
        from wtss.data.capsule import load_capsule_synthetic_cohort

        return load_capsule_synthetic_cohort(size)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="isic2018")
    ap.add_argument("--backbone", default="dino518")
    ap.add_argument("--phase", default="corr", choices=["corr", "occlusion"])
    ap.add_argument("--artifact", default="ruler_fixed")
    ap.add_argument("--arms", nargs="+", default=["erm", "mask"])
    ap.add_argument("--overlaps", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--proposed", action="store_true", help="add the proposed localisation-free arms")
    ap.add_argument("--analyse_only", action="store_true")
    a = ap.parse_args()

    from wtss.backbones import BACKEND_SIZE

    size = BACKEND_SIZE[a.backbone]
    cohort, placements, loaders, geometry = cohort_and_placements(a.cohort, size)
    out = paths.RESULTS / "synthetic" / a.cohort / f"{a.backbone}_{a.artifact}_{a.phase}_{a.tag}"
    cfg = SynthConfig(phase=a.phase, overlaps=a.overlaps, seeds=a.seeds, arms=a.arms, artifact=a.artifact,
                      batch_size=a.batch_size, workers=a.workers, n_boot=a.n_boot)
    if geometry:
        cfg.geometry = geometry
    if a.proposed:
        from wtss.methods.insertion import synthetic_extra_arms

        cfg.extra_arms = synthetic_extra_arms(cfg)
    if not a.analyse_only:
        run_synthetic(cohort, a.backbone, placements, out, cfg, paths.CACHE / "images", loaders,
                      paths.CACHE / "features", torch.device("cuda"))
    boot = analyse_synthetic(out, n_boot=a.n_boot)
    print(boot[["overlap", "method_a", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
