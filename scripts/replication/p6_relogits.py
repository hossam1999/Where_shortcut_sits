"""Re-extract the per-image test logits of every finished P6 run in the CSV's row order (fix of 2026-11-20: p6_epvt_score.py
saved them from their FastDataLoader, which samples in random order, so the saved files were misaligned with the image ids;
the registered AUROCs, computed by their eval_indomain, were correct). Same checkpoints, same test images; each file must
reproduce its run's registered AUROC (assert), so nothing is re-scored. Misaligned files kept as *_misaligned.csv.gz.
    EPVT_* env as for scoring; /workspace/envs/epvt/bin/python scripts/replication/p6_relogits.py"""
import glob, json, os, sys
sys.path.insert(0, "/workspace/third_party/epvt")
import pandas as pd, torch
from sklearn.metrics import roc_auc_score
from domainbed import algorithms, datasets
D = os.environ.get("WTSS_DATA", "/workspace/data")
for js in sorted(glob.glob(f"{D}/epvt/runs/*/*__test_bias_*.json")):
    j = json.load(open(js)); base = js[:-5]; csv = base + ".csv.gz"
    if os.path.exists(base + "_misaligned.csv.gz"):
        continue
    ck = os.path.join(os.path.dirname(js), os.path.basename(j["ckpt"]))   # the run folder may have been renamed (preA33_*)
    c = torch.load(ck, map_location="cpu", weights_only=False)
    alg = algorithms.get_algorithm_class(c["args"]["algorithm"])(c["model_input_shape"], c["model_num_classes"], 5, c["model_hparams"])
    alg.load_state_dict(c["model_dict"]); alg.cuda().eval()
    df = pd.read_csv(os.environ["EPVT_EVAL_ROOT"] + j["test"])
    ds = datasets.MelanomaDataset(df=df, imfolder=os.environ["EPVT_IMFOLDER"]); L = []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=8):
            L.append(alg.predict(x.cuda(), domain=None).float().cpu())
    L = torch.cat(L).numpy(); auc = roc_auc_score(df.label, L[:, 1] - L[:, 0])
    assert abs(auc - j["roc_auc"]) < 1e-4, (js, auc, j["roc_auc"])
    os.rename(csv, base + "_misaligned.csv.gz")
    pd.DataFrame({"image": df.image, "label": df.label, "logit0": L[:, 0], "logit1": L[:, 1]}).to_csv(csv, index=False)
    print(f"{os.path.basename(os.path.dirname(js))}: AUROC from re-extracted logits {auc:.6f} = registered {j['roc_auc']:.6f}", flush=True)
    del alg; torch.cuda.empty_cache()
