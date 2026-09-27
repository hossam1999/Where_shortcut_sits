"""U7: U-MtE − mte_aug (erase vs augment with the same generic overlays), reversed AUROC, both traps."""
import json

import pandas as pd

from wtss import paths
from wtss.stats import hierarchical_paired_bootstrap, slim

out = {}
for name, p in (("thyroid", paths.RESULTS / "thyroid" / "dino518_ablation"),
                ("isic2019", paths.RESULTS / "spec_e13" / "dino518_spec_ablation")):
    preds = pd.read_csv(p / "predictions.csv.gz")
    for trap in ("trapA", "trapB"):
        for a1, a0 in (("mte", "mte_aug"), ("mte_aug", "mask"), ("insert_aug", "erm")):
            for env in ("test_rev", "clean"):
                r = hierarchical_paired_bootstrap(slim(preds[preds.trap == trap], env, (a1, a0)), a1, a0, env, 10000, 5, fast=True)
                out[f"{name}|{trap}|{a1}-{a0}|{env}"] = [round(r[k], 3) for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")]
                print(name, trap, f"{a1}-{a0}", env, out[f"{name}|{trap}|{a1}-{a0}|{env}"], flush=True)
(paths.RESULTS / "ablation_u7.json").write_text(json.dumps(out, indent=1))
