"""P6 (EPVT, MICCAI 2023): score one trained model of their code on a Bissoto ISIC 2019 trap test set, in-domain only.
Their test_erm.py / test_epvt.py first evaluate the four OOD sets (not needed here: the P6 comparison is the trap sets)
and then the in-domain ISIC test exactly like this: MelanomaDataset (Resize 224, ImageNet normalisation), the algorithm
rebuilt from the checkpoint's own hparams, misc.eval_indomain -> metrics.compute_isic_metrics_binary (their ROC-AUC).
Per-image logits are saved for the paired comparisons later (nothing of ours is scored here).

    EPVT_* env as for training; python scripts/replication/p6_epvt_score.py --ckpt <run>/<exp>.pkl --test test_bias_1_1.csv
"""
import argparse, json, os, sys
sys.path.insert(0, "/workspace/third_party/epvt")
import numpy as np, pandas as pd, torch
from domainbed import algorithms, datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--test", required=True)
a = ap.parse_args()
c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
# their test_epvt.py / test_erm.py build the algorithm with 5 domains (the SKIN environments C, D, G, H, R); the checkpoint's
# "model_num_domains" (= len(dataset) - len(test_envs) = 4) does not match the trained prompt tensors, so it is not used
alg = algorithms.get_algorithm_class(c["args"]["algorithm"])(c["model_input_shape"], c["model_num_classes"], 5, c["model_hparams"])
alg.load_state_dict(c["model_dict"]); alg.to("cuda" if torch.cuda.is_available() else "cpu")
dev = next(alg.parameters()).device
df = pd.read_csv(os.environ["EPVT_EVAL_ROOT"] + a.test)
ds = datasets.MelanomaDataset(df=df, imfolder=os.environ["EPVT_IMFOLDER"])
ld = FastDataLoader(dataset=ds, batch_size=128, num_workers=8)
ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, _ = misc.eval_indomain(alg, loader=ld, valid_df=df, batch_size=128, weights=None,
                                                                   device=dev, name=None, domain=None)
alg.eval(); L = []
with torch.no_grad():
    for x, _ in FastDataLoader(dataset=ds, batch_size=128, num_workers=8):
        L.append(alg.predict(x.to(dev), domain=None).float().cpu())
L = torch.cat(L).numpy()
out = os.path.splitext(a.ckpt)[0] + f"__{os.path.splitext(a.test)[0]}"
pd.DataFrame({"image": df.image, "label": df.label, "logit0": L[:, 0], "logit1": L[:, 1]}).to_csv(out + ".csv.gz", index=False)
res = {"ckpt": a.ckpt, "test": a.test, "n": len(df), "roc_auc": float(AUC), "acc": float(ACC), "bacc": float(BACC), "f1": float(F1)}
open(out + ".json", "w").write(json.dumps(res)); print(json.dumps(res))
