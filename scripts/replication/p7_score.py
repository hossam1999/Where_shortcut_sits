"""P7 (RR-ClArC, A28): score our RR-ClArC fine-tune of one architecture once, exactly as p6_step1.py scored the released
Vanilla checkpoints (their dataset classes, split default_rng(0) 80/10/10, LSB variants clean / attacked (every image of every
class = the paper's "biased" test) / ch (as the fine-tune data), utils/metrics accuracy and macro-F1), test and val.
Per-image logits are kept for the paired comparisons (A28). Outputs (pushed): $WTSS_DATA/results_push/results/replication/
rrclarc/p7/<arch>_{summary.csv, logits_<split>_<variant>.csv.gz}.

    cd /workspace/third_party/rrclarc && /workspace/envs/r2r/bin/python <repo>/scripts/replication/p7_score.py <arch>
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
RC = "/workspace/third_party/rrclarc"; sys.path.insert(0, RC)
from torch.utils.data import DataLoader
from datasets import get_dataset
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.metrics import compute_metrics
from experiments.evaluation.evaluate_by_subset_attacked import compute_model_scores
D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = Path(f"{D}/results_push/results/replication/rrclarc/p7"); OUT.mkdir(parents=True, exist_ok=True)
torch.random.manual_seed(0)


def main(arch):
    name = f"{arch}_RRClarc_p7"
    cfg = yaml.safe_load(open(f"{RC}/config_files/correcting_isic_attacked/local/{name}.yaml"))
    ak = get_artifact_kwargs(cfg); mk = get_dataset(cfg["dataset_name"])
    ds = {"ch": mk(data_paths=cfg["data_paths"], normalize_data=True, p_artifact=cfg["p_artifact"],
                   artifact_type=cfg["artifact_type"], attacked_classes=cfg["attacked_classes"], **ak)}
    ds["clean"] = mk(data_paths=cfg["data_paths"], normalize_data=True, attacked_classes=[], **ak)
    ds["attacked"] = mk(data_paths=cfg["data_paths"], normalize_data=True, p_artifact=1.0,
                        artifact_type=cfg["artifact_type"], attacked_classes=ds["ch"].class_names, **ak)
    base = ds["ch"]
    model = get_fn_model_loader(arch)(n_class=len(base.class_names), ckpt_path=f"{RC}/{cfg['ckpt_path_corrected']}").to("cuda").eval()
    rows = []
    for split, idx in (("test", base.idxs_test), ("val", base.idxs_val)):
        for var in ("clean", "attacked", "ch"):
            outs, y = compute_model_scores(model, DataLoader(ds[var].get_subset_by_idxs(idx), batch_size=32, shuffle=False, num_workers=8), "cuda")
            m = compute_metrics(outs, y, None)
            pd.DataFrame(outs.numpy(), columns=[f"logit_{c}" for c in base.class_names]).assign(idx=np.asarray(idx), y=y.numpy()) \
              .to_csv(OUT / f"{arch}_logits_{split}_{var}.csv.gz", index=False)
            rows.append({"arch": arch, "lamb": cfg["lamb"], "layer": cfg["layer_name"], "p_artifact": cfg["p_artifact"],
                         "split": split, "variant": var, "n": len(y), "accuracy": m["accuracy"], "f1_macro": m["f1"]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"{arch}_summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
