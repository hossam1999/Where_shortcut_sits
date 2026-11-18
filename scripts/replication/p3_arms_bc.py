"""P3 (Bassi) 90/10 inside variant (A6): our arms b (DINOv2 ViT-B/14 @ 518 on lung-masked images + mask_cmc) and c (U-MtE,
lung ROI, label-free), on exactly the data of the ISNet run `isnet_inside_9010_seed0` (armd_train.P3: P3's cohort cache,
marks inside the lungs on 90% of training/validation images keyed by data seed 0; tests biased / standard / deceiving on
the rebuilt test, both A9 variants). Images enter without augmentation; the lung ROI (PSPNet, P3's cache) is applied by
U-MtE's masked view (outside = mean colour), as for P2's arms b, c.
Heads (section 3, registered grid C in {0.01, 0.1, 1, 10}; validation = macro-F1 on P3's validation split with its marks):
  b: mask_cmc (mode pooled / per_class, chosen on validation; artifact label = mark present) + class-balanced multinomial LR;
  c: U-MtE eraser (difference subspace of masked vs generic-overlay training features) + class-balanced multinomial LR.

    python scripts/replication/p3_arms_bc.py --stage features     # GPU
    python scripts/replication/p3_arms_bc.py --stage heads        # CPU
Writes $WTSS_DATA/replication/isnet/arms_bc/ and results/replication/isnet/p3_9010_inside_arms_bc.{md,csv}.
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
OUT = Path(f"{D}/replication/isnet/arms_bc")
RES = ROOT / "results/replication/isnet"
CS = (0.01, 0.1, 1.0, 10.0)


def features():
    from PIL import Image
    import armd_train as A
    from wtss.umte import UMtE
    S = A.P3()
    OUT.mkdir(parents=True, exist_ok=True)
    um = UMtE("dino518", device="cuda", seed=0, batch_size=16)
    jobs = [(k, False) for k in S.sets] + [("train", True)]
    for k, ins in jobs:
        f = OUT / f"dino518_{'insert' if ins else 'masked'}_{k}.npz"
        if f.exists():
            continue
        g, r, idx, ids, y, pl, marks = S.src[k]
        data = S.T.Data(g, r, ids, y, pl, marks, False, 0)
        X = []
        for s0 in range(0, len(ids), 512):
            ims, rois = [], []
            for j in range(s0, min(s0 + 512, len(ids))):
                gi, mi, _ = data[j]
                ims.append(Image.fromarray((gi[0].numpy() * 255).round().astype(np.uint8)).convert("RGB"))
                rois.append(mi[0].numpy() > 0.5)
            X.append(um.masked_features(ims, rois, insert=ins, keys=ids[s0:s0 + 512] if ins else None))
            print(k, "insert" if ins else "masked", s0 + len(ims), "/", len(ids), flush=True)
        np.savez(f, ids=np.array(ids, dtype="U"), X=np.concatenate(X))


def heads():
    import armd_train as A
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from wtss.methods.insertion import fit_difference_subspace
    S = A.P3()
    X = {k: np.load(OUT / f"dino518_masked_{k}.npz")["X"] for k in S.sets}
    X1 = np.load(OUT / "dino518_insert_train.npz")["X"]
    ytr, atr = S.labels("train"); yv = S.labels("valid")[0]
    tests = [k for k in S.sets if k.startswith("test_")]
    out_rows, md = [], ["# P3 90/10 inside variant (A6): our arms b and c, seed 0 data (= the ISNet 90/10 inside run's data)", "",
                        "Macro-F1 on the rebuilt P3 test (both A9 variants). P3 is descriptive only (A9 check failed, "
                        "2026-11-18). One data seed; point estimates; the paired comparison with ISNet is added when that run "
                        "is scored. Validation = P3's validation split with its marks (registered); registered C grid.", ""]

    def fit(Z, label):
        best = None
        for C in CS:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=0).fit(Z["train"], ytr)
            v = float(f1_score(yv, clf.predict(Z["valid"]), average="macro"))
            if best is None or v > best[1]:
                best = (clf, v, C)
        return best

    arms = {}
    best_b = None
    for mode in CA.CMC_MODES:
        P = CA.Projector(CA.cmc_directions(X["train"], ytr, atr, mode))
        Z = {k: P(v) for k, v in X.items()}
        clf, v, C = fit(Z, mode)
        if best_b is None or v > best_b[1]:
            best_b = (clf, v, C, mode, Z)
    arms["b_dino518_mask_cmc"] = best_b
    er = fit_difference_subspace(X["train"], X1, energy=0.90, max_k=64, seed=0)
    Zc = {k: er(v) for k, v in X.items()}
    clf, v, C = fit(Zc, "umte")
    arms["c_umte"] = (clf, v, C, f"k = {er.k}", Zc)
    preds = []
    for name, (clf, v, C, mode, Z) in arms.items():
        md.append(f"- **{name}**: selected mode **{mode}**, C = {C:g} (validation macro-F1 {v:.3f}; edge of the registered grid: "
                  f"{'lowest' if C == CS[0] else 'highest' if C == CS[-1] else 'no'})")
        for k in tests:
            d = S.part[k]; y = d.y.to_numpy(); pr = clf.predict_proba(Z[k])
            preds.append(pd.DataFrame({"image_id": d.image_id, "y": y, "method": name, "test_set": k, "size_matched": d.size_matched,
                                       **{f"p{j}": pr[:, j] for j in range(3)}}))
            for var, keep in (("primary", np.ones(len(y), bool)), ("size-matched", d.size_matched.to_numpy().astype(bool))):
                out_rows.append({"arm": name, "mode": mode, "C": C, "test": k[5:], "variant": var, "n": int(keep.sum()),
                                 "macro_f1": float(f1_score(y[keep], pr[keep].argmax(1), average="macro"))})
    pd.concat(preds).to_csv(OUT / "preds_test.csv.gz", index=False)
    res = pd.DataFrame(out_rows); RES.mkdir(parents=True, exist_ok=True); res.to_csv(RES / "p3_9010_inside_arms_bc.csv", index=False)
    md += ["", "| arm | test | primary | size-matched |", "|---|---|---|---|"]
    for (arm, t), g in res.groupby(["arm", "test"], sort=False):
        md.append(f"| {arm} | {t} | {g[g.variant == 'primary'].macro_f1.iloc[0]:.3f} | {g[g.variant == 'size-matched'].macro_f1.iloc[0]:.3f} |")
    (RES / "p3_9010_inside_arms_bc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "heads"], required=True)
    features() if ap.parse_args().stage == "features" else heads()
