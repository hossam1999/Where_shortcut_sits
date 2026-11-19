"""P4 (Saab, ISIC only, A13): our arms b (DINOv2 ViT-B/14 @ 518 on lesion-masked images + mask_cmc) and c (U-MtE, lesion
ROI, label-free), Bissoto trap sets 1-3 (train 1,822 / validation 259 / trap test 772). Images and Task 1 lesion masks
from p4_train's 512-px cache (ROI = ISIC 2018 Task 1 lesion mask, A13 item 8); no augmentation; U-MtE's masked view
(outside = mean colour). Artifact labels (arm b only): every Bissoto flag with >= 3 images in each artifact x diagnosis
group in that split's training set (A13 item 8 as read on 2026-11-18). Heads: class-balanced logistic regression,
registered C grid {0.01, 0.1, 1, 10}; arm b's mode (pooled / per_class) and C by validation AUROC. Robust AUROC on the
trap test, scored once per split; compared with our Binary-ERM and Segmentation-ERM of the same split (descriptive until
the P4 gate).

    python scripts/replication/p4_arms_bc.py --stage features     # GPU (all 2,594 images once)
    python scripts/replication/p4_arms_bc.py --stage heads        # CPU
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
OUT = Path(f"{D}/replication/saab/arms_bc")
RES = ROOT / "results/replication/saab"
CS = (0.01, 0.1, 1.0, 10.0)
FLAGS = ("dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches", "vasc")


def rows(k):
    import p4_train as T
    r = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(f"{T.B}/isic_annotated_{s}{k}.csv").rename(columns={"label": "target"})
        d["id"] = d.image.str.split(".").str[0]
        r[s] = d.reset_index(drop=True)
    return r


def features():
    from PIL import Image
    import p4_train as T
    from wtss.umte import UMtE
    allr = [rows(k) for k in (1, 2, 3)]
    img, msk, ids = T.cache([i for r in allr for s in r for i in r[s].id])
    OUT.mkdir(parents=True, exist_ok=True)
    um = UMtE("dino518", device="cuda", seed=0, batch_size=16)
    f = OUT / "dino518_masked_all.npz"
    if not f.exists():
        X = []
        for s0 in range(0, len(ids), 256):
            ims = [Image.fromarray(np.asarray(img[j])) for j in range(s0, min(s0 + 256, len(ids)))]
            X.append(um.masked_features(ims, [np.asarray(msk[j]) > 0 for j in range(s0, min(s0 + 256, len(ids)))]))
            print("masked", s0 + len(ims), "/", len(ids), flush=True)
        np.savez(f, ids=np.array(ids, dtype="U"), X=np.concatenate(X))
    for k in (1, 2, 3):   # U-MtE's overlay view of each split's training images
        g = OUT / f"dino518_insert_train{k}.npz"
        if g.exists():
            continue
        tid = allr[k - 1]["train"].id.tolist(); pos = {i: j for j, i in enumerate(ids)}
        X = []
        for s0 in range(0, len(tid), 256):
            js = [pos[i] for i in tid[s0:s0 + 256]]
            X.append(um.masked_features([Image.fromarray(np.asarray(img[j])) for j in js], [np.asarray(msk[j]) > 0 for j in js],
                                        insert=True, keys=tid[s0:s0 + 256]))
        np.savez(g, ids=np.array(tid, dtype="U"), X=np.concatenate(X))
        print("insert train", k, flush=True)


def heads():
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from wtss.methods.insertion import fit_difference_subspace
    from wtss.replication_stats import paired_bootstrap
    z = np.load(OUT / "dino518_masked_all.npz"); pos = {i: j for j, i in enumerate(z["ids"].astype(str))}; XA = z["X"]
    res, preds, md = [], [], ["# P4 (Saab, ISIC, A13): our arms b and c on Bissoto trap sets 1-3", "",
                              "Robust AUROC on the 772-image trap test per split; image-only paired bootstrap per split (one network "
                              "per split). Descriptive until the P4 gate (A13). Validation (259 images) lies inside training "
                              "(Bissoto's trap sets, A13), so validation-based choices say little.", ""]
    for k in (1, 2, 3):
        r = rows(k)
        X = {s: XA[[pos[i] for i in r[s].id]] for s in r}
        y = {s: r[s].target.to_numpy() for s in r}
        tr = r["train"]
        used = [f for f in FLAGS if (pd.crosstab(tr[f], tr.target).reindex(index=[0, 1], columns=[0, 1], fill_value=0) >= 3).all().all()]
        A = tr[used].to_numpy().astype(int)

        def fitc(Z):
            best = None
            for C in CS:
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=0).fit(Z["train"], y["train"])
                v = roc_auc_score(y["val"], clf.decision_function(Z["val"]))
                if best is None or v > best[1]:
                    best = (clf, v, C)
            return best
        bb = None
        for mode in CA.CMC_MODES:
            Q = np.concatenate([CA.cmc_directions(X["train"], y["train"], A[:, j], mode) for j in range(A.shape[1])], 1)
            q, rr = np.linalg.qr(Q); Q = q[:, np.abs(np.diag(rr)) > 1e-10 * max(1.0, np.abs(rr).max())]
            P = CA.Projector(Q); Z = {s: P(v) for s, v in X.items()}
            clf, v, C = fitc(Z)
            if bb is None or v > bb[1]:
                bb = (clf, v, C, mode, Z)
        X1 = np.load(OUT / f"dino518_insert_train{k}.npz")["X"]
        er = fit_difference_subspace(X["train"], X1, energy=0.90, max_k=64, seed=0)
        Zc = {s: er(v) for s, v in X.items()}
        cc = fitc(Zc)
        refs = {}
        for name, run in (("Binary-ERM", f"erm_split{k}"), ("Seg-ERM", f"seg_split{k}")):
            f = Path(f"{D}/replication/saab/runs/{run}/preds_test.csv")
            if f.exists():
                d = pd.read_csv(f); refs[name] = d.set_index(d.id.astype(str)).loc[r["test"].id, "score"].to_numpy()
        for arm, (clf, v, C, mode, Z) in (("b_dino518_mask_cmc", bb), ("c_umte", (*cc, f"k = {er.k}", Zc))):
            sc = clf.decision_function(Z["test"]); auc = roc_auc_score(y["test"], sc)
            row = {"split": k, "arm": arm, "mode": mode, "C": C, "val_auroc": v, "robust_auroc": auc, "flags": ",".join(used)}
            for n, ref in refs.items():
                b = paired_bootstrap({0: (r["test"].id.tolist(), y["test"], sc, ref)}, "auroc")
                row[f"vs_{n}"] = f"{b['delta']:+.3f} [{b['ci95_lo']:+.3f}, {b['ci95_hi']:+.3f}]"
                row[f"{n}_auroc"] = roc_auc_score(y["test"], ref)
            res.append(row)
            preds.append(pd.DataFrame({"image_id": r["test"].id, "label": y["test"], "method": arm, "split": k, "score": sc}))
    res = pd.DataFrame(res); RES.mkdir(parents=True, exist_ok=True); res.to_csv(RES / "p4_arms_bc.csv", index=False)
    pd.concat(preds).to_csv(OUT / "preds_test.csv.gz", index=False)
    cols = [c for c in res.columns if c.startswith("vs_")]
    md += ["| split | arm | mode | C | robust AUROC | " + " | ".join(f"Δ {c[3:]} [95% CI]" for c in cols) + " |",
           "|---|---|---|---|---|" + "---|" * len(cols)]
    md += [f"| {x.split} | {x.arm} | {x['mode']} | {x.C:g} | {x.robust_auroc:.3f} | " + " | ".join(str(x[c]) for c in cols) + " |" for _, x in res.iterrows()]
    md += ["", "Mean over splits: " + ", ".join(f"{a} {g.robust_auroc.mean():.3f} ± {g.robust_auroc.std(ddof=1):.3f}" for a, g in res.groupby("arm"))]
    (RES / "p4_arms_bc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "heads"], required=True)
    features() if ap.parse_args().stage == "features" else heads()
