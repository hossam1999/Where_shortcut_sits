"""Zero-shot reliance of vision–language backbones on in-ROI artifacts (docs/PREREGISTRATION_TEXT_PROMPT.md, Z1).

No training: score = <x, c_pos − c_neg> with class-prompt embeddings (results/text_dirs/*.npz), on cached joint-space
image features (unmasked 'erm' and masked views) of the same deterministic trap environments as the trained arms.
  python scripts/analysis/zero_shot.py      # -> results/zero_shot/summary.csv, boot.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths
from wtss.data.isic2019_spec import build_spec_envs, load_spec_cohort
from wtss.stats import hierarchical_paired_bootstrap, safe_auc

spec = importlib.util.spec_from_file_location("rt", Path(__file__).resolve().parents[1] / "run_thyroid_traps.py")
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
F = paths.CACHE / "features"
JOBS = [("medsiglip448", "thyroid", F / "thyroid" / "medsiglip_448"), ("medsiglip448", "capsule", F / "capsule" / "medsiglip_448"),
        ("medsiglip448", "ovary", F / "ovary" / "medsiglip_448"), ("dermlip224", "isic", F / "spec_isic2019" / "dermlip_panderm_224")]


def envs_for(cohort):
    if cohort == "isic":
        return build_spec_envs(load_spec_cohort())
    c = {"thyroid": rt.cohort, "capsule": rt.capsule_cohort, "ovary": rt.ovary_cohort}[cohort]()
    return build_spec_envs(c, group_col="group")


def main():
    out = paths.ensure(paths.RESULTS / "zero_shot")
    rows, boots = [], {}
    for bb, cohort, fdir in JOBS:
        td = paths.RESULTS / "text_dirs" / f"{bb}_{cohort}.npz"
        if not td.exists() or not (fdir / "erm.npz").exists():
            print("skip", bb, cohort); continue
        c = np.load(td)["class_emb"]; w = c[1] - c[0]
        S = {}
        for view in ("erm", "mask"):
            z = np.load(fdir / f"{view}.npz"); S[view] = dict(zip(z["ids"].astype(str), z["X"] @ w))
        envs = envs_for(cohort)
        frames = []
        for (trap, seed, k, env), d in envs.items():
            if env not in ("test_corr", "test_rev", "clean"):
                continue
            for view, name in (("erm", "zs"), ("mask", "zs_mask")):
                frames.append(pd.DataFrame({"trap": trap, "seed": seed, "fold": k, "env": env, "method": name,
                                            "image_id": d.image_id.astype(str).values, "y": d.y.values,
                                            "prob": [S[view][i] for i in d.image_id.astype(str)]}))
        P = pd.concat(frames, ignore_index=True)
        auc = P.groupby(["trap", "method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
        m = auc.groupby(["trap", "method", "env"]).mean().unstack().round(3)
        m["corr_minus_rev"] = (m.test_corr - m.test_rev).round(3)
        print(f"== {bb} {cohort}\n{m.to_string()}", flush=True)
        for (trap, meth), r in m.iterrows():
            rows.append({"backbone": bb, "cohort": cohort, "trap": trap, "method": meth, **r.to_dict()})
        for trap in ("trapA", "trapB"):
            q = P[P.trap == trap]
            r = hierarchical_paired_bootstrap(q, "zs_mask", "zs", "test_rev", 10000, 3, fast=True)
            boots[f"{bb}|{cohort}|{trap}|zs_mask-zs"] = [round(r[x], 3) for x in ("seed_delta_mean", "ci95_lo", "ci95_hi")]
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (out / "boot.json").write_text(json.dumps(boots, indent=1))
    print(json.dumps(boots, indent=1))


if __name__ == "__main__":
    main()
