"""End-to-end fine-tuning robustness check on the real traps.

  python scripts/run_finetune.py --cohort isic2019 --arch resnet50
  python scripts/run_finetune.py --cohort nih_drain --arch resnet50
"""
from __future__ import annotations

import argparse

import torch

from wtss import paths
from wtss.experiments.finetune import run_finetune
from wtss.experiments.real_traps import make_renderers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="isic2019", choices=["isic2019", "nih_drain"])
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--arms", nargs="+", default=["erm", "mask", "balanced", "insert_aug", "i2e_ft"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    import run_traps as rt  # noqa: E402  (same cohort builders)

    args = argparse.Namespace(exclude_vignetting=False)
    df, envs, counts, donors, cache, insert_fn = {"isic2019": rt.isic2019, "nih_drain": rt.nih_drain}[a.cohort](args)
    render = make_renderers(cache, 518, donors, insert_fn)
    traps = ("trapA", "trapB") if a.cohort == "isic2019" else ("drain",)
    out = paths.RESULTS / "finetune" / a.cohort / a.arch
    run_finetune(a.cohort, envs, render, out, a.arms, traps, a.folds, arch=a.arch, epochs=a.epochs,
                 device=torch.device("cuda"))
    from wtss.experiments.real_traps import analyse_traps

    res = analyse_traps(out)
    b = res["boot"]
    print(b[(b.env == "test_rev")][["trap", "arm", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
