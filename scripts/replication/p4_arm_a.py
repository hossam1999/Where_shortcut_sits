"""P4 (Saab, ISIC only, A13 item 8a): our arm a = mask_cmc on THEIR trained Segmentation-ERM encoder, per split k.

Features: the SE-ResNeXt50 (32x4d) encoder of our reproduced Segmentation-ERM split k (best.pt, the validation-Dice
checkpoint scored on the trap test), last encoder stage (2,048 channels, 16 x 16 at 512 px), global average pooled, on
**lesion-masked** images: p4_train's 512-px cache, outside the ISIC 2018 Task 1 lesion mask replaced by the mean colour
(wtss.utils.MEAN_RGB, as arm b's masking), then their ISIC normalisation. No augmentation.
Head: identical to arm b (p4_arms_bc.py): artifact flags with >= 3 training images in each artifact x diagnosis group of
split k (A13 item 8 as read on 2026-11-18), their within-class mean differences projected out (mode pooled / per_class),
class-balanced logistic regression, registered C grid {0.01, 0.1, 1, 10} (A21: unchanged for arm a), mode and C by
validation AUROC. Robust AUROC on the 772-image trap test, scored once; paired image-only bootstrap vs Seg-ERM split k
(and Binary-ERM split k) on the same test images. Descriptive until the P4 gate (A13).

    python scripts/replication/p4_arm_a.py --stage features --split k    # GPU (saab env)
    python scripts/replication/p4_arm_a.py --stage heads                 # CPU (our env), every split with features
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = Path(f"{D}/replication/saab/arm_a")
RES = ROOT / "results/replication/saab"
CS = (0.01, 0.1, 1.0, 10.0)
MEAN_RGB = (124, 116, 104)   # wtss.utils.MEAN_RGB (checked in heads(); the saab env does not import wtss)


def features(k):
    import torch
    import p4_train as T
    from p4_arms_bc import rows
    r = rows(k)
    ids_k = [i for s in r for i in r[s].id]
    img, msk, ids = T.cache(ids_k)
    pos = {i: j for j, i in enumerate(ids)}
    model, _ = T.build("seg")
    T.add_checkpointing(model)            # same module names as the saved state dict (Ckpt is inactive in eval mode)
    c = torch.load(f"{D}/replication/saab/runs/seg_split{k}/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(c["model"]); model.cuda().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    X = []
    with torch.no_grad():
        for s0 in range(0, len(ids_k), 32):
            xs = []
            for i in ids_k[s0:s0 + 32]:
                x = np.asarray(img[pos[i]]).copy()
                x[np.asarray(msk[pos[i]]) == 0] = np.asarray(MEAN_RGB, np.uint8)
                xs.append(((x.astype(np.float32) / 255.0 - T.MEAN) / T.STD).transpose(2, 0, 1))
            f = model.encoder(torch.from_numpy(np.stack(xs)).cuda())[-1]
            X.append(f.mean((2, 3)).float().cpu().numpy())
    np.savez(OUT / f"seg_split{k}_encoder_masked.npz", ids=np.array(ids_k, dtype="U"), X=np.concatenate(X),
             best_epoch=c["epoch"])
    print(f"split {k}: {len(ids_k)} feature rows from best epoch {c['epoch']}", flush=True)


def heads():
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from wtss.utils import MEAN_RGB as M
    from wtss.replication_stats import paired_bootstrap
    from p4_arms_bc import FLAGS, rows
    assert tuple(M) == MEAN_RGB, (M, MEAN_RGB)
    res, preds = [], []
    md = ["# P4 (Saab, ISIC, A13 item 8a): arm a = mask_cmc on their Segmentation-ERM encoder (machine 2, A27)", "",
          "Lesion-masked images through our reproduced Seg-ERM's SE-ResNeXt50 encoder (best validation-Dice epoch), pooled "
          "2,048-d features, mask_cmc + class-balanced logistic regression (C grid {0.01, 0.1, 1, 10}, mode and C by "
          "validation AUROC). Robust AUROC on the 772-image trap test, scored once per split; image-only paired bootstrap "
          "vs Seg-ERM of the same split (same test images). Descriptive until the P4 gate (A13). Validation (259 images) "
          "lies inside training (Bissoto's trap sets, A13), so validation-based choices say little.", ""]
    for k in (1, 2, 3):
        f = OUT / f"seg_split{k}_encoder_masked.npz"
        if not f.exists():
            continue
        z = np.load(f); pos = {i: j for j, i in enumerate(z["ids"].astype(str))}
        r = rows(k)
        X = {s: z["X"][[pos[i] for i in r[s].id]] for s in r}
        y = {s: r[s].target.to_numpy() for s in r}
        tr = r["train"]
        used = [g for g in FLAGS if (pd.crosstab(tr[g], tr.target).reindex(index=[0, 1], columns=[0, 1], fill_value=0) >= 3).all().all()]
        A = tr[used].to_numpy().astype(int)
        bb = None
        for mode in CA.CMC_MODES:
            Q = np.concatenate([CA.cmc_directions(X["train"], y["train"], A[:, j], mode) for j in range(A.shape[1])], 1)
            q, rr = np.linalg.qr(Q); Q = q[:, np.abs(np.diag(rr)) > 1e-10 * max(1.0, np.abs(rr).max())]
            P = CA.Projector(Q); Z = {s: P(v) for s, v in X.items()}
            for C in CS:
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=0).fit(Z["train"], y["train"])
                v = roc_auc_score(y["val"], clf.decision_function(Z["val"]))
                if bb is None or v > bb[1]:
                    bb = (clf, v, C, mode, Z)
        clf, v, C, mode, Z = bb
        sc = clf.decision_function(Z["test"]); auc = roc_auc_score(y["test"], sc)
        row = {"split": k, "arm": "a_segerm_mask_cmc", "best_epoch": int(z["best_epoch"]), "mode": mode, "C": C,
               "C_edge": "lowest" if C == CS[0] else "highest" if C == CS[-1] else "", "val_auroc": v,
               "robust_auroc": auc, "flags": ",".join(used)}
        for n, run in (("Seg-ERM", f"seg_split{k}"), ("Binary-ERM", f"erm_split{k}")):
            p = Path(f"{D}/replication/saab/runs/{run}/preds_test.csv")
            if not p.exists():
                p = ROOT / f"results/replication/saab/scores/{run}.csv.gz"   # Binary-ERM: machine 1's scores in git
            if p.exists():
                d = pd.read_csv(p); ref = d.set_index(d.id.astype(str)).loc[r["test"].id, "score"].to_numpy()
                b = paired_bootstrap({0: (r["test"].id.tolist(), y["test"], sc, ref)}, "auroc")
                row[f"{n}_auroc"] = roc_auc_score(y["test"], ref)
                row[f"vs_{n}"] = f"{b['delta']:+.3f} [{b['ci95_lo']:+.3f}, {b['ci95_hi']:+.3f}]"
        res.append(row)
        preds.append(pd.DataFrame({"image_id": r["test"].id, "label": y["test"], "method": row["arm"], "split": k, "score": sc}))
    res = pd.DataFrame(res); RES.mkdir(parents=True, exist_ok=True); res.to_csv(RES / "p4_arm_a.csv", index=False)
    (RES / "scores").mkdir(exist_ok=True)
    pd.concat(preds).to_csv(RES / "scores/p4_arm_a.csv.gz", index=False)
    cols = [c for c in res.columns if c.startswith("vs_")]
    md += ["| split | best epoch | mode | C (edge) | robust AUROC | Seg-ERM | " + " | ".join(f"Δ {c[3:]} [95% CI]" for c in cols) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(cols)]
    md += [f"| {x.split} | {x.best_epoch} | {x['mode']} | {x.C:g} {('(' + x.C_edge + ')') if x.C_edge else ''} | "
           f"{x.robust_auroc:.3f} | {x.get('Seg-ERM_auroc', float('nan')):.3f} | " + " | ".join(str(x[c]) for c in cols) + " |"
           for _, x in res.iterrows()]
    if len(res) > 1:
        md += ["", f"Mean over {len(res)} splits: arm a {res.robust_auroc.mean():.3f} ± {res.robust_auroc.std(ddof=1):.3f}"
               + (f", Seg-ERM {res['Seg-ERM_auroc'].mean():.3f} ± {res['Seg-ERM_auroc'].std(ddof=1):.3f}" if "Seg-ERM_auroc" in res else "")]
    (RES / "p4_arm_a.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "heads"], required=True)
    ap.add_argument("--split", type=int)
    a = ap.parse_args()
    features(a.split) if a.stage == "features" else heads()
