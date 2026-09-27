"""Compute text-prompted artifact subspaces + class-prompt embeddings once per (backbone, cohort).
  python scripts/make_text_directions.py        # -> results/text_dirs/{backbone}_{cohort}.npz
"""
import numpy as np

from wtss import paths
from wtss.text_directions import artifact_subspace

JOBS = [("medsiglip448", "thyroid"), ("medsiglip448", "capsule"), ("medsiglip448", "ovary"), ("dermlip224", "isic")]
if __name__ == "__main__":
    out = paths.ensure(paths.RESULTS / "text_dirs")
    for bb, c in JOBS:
        r = artifact_subspace(bb, c)
        np.savez(out / f"{bb}_{c}.npz", **r)
        print(bb, c, "k =", int(r["k"]))
