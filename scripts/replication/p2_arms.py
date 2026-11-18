"""P2 (Pahde 2025) — our arms on ISIC 2019 (section 3 of docs/PREREGISTRATION_REPLICATION.md). CPU heads only.

  a) mask_cmc on our Vanilla ResNet-50d (their trainer, seed 0): penultimate features of the unmasked images as their
     network sees them (saved by p2_score.py --features), the within-class artifact mean difference projected out
     (`cmc_directions` + `Projector`; only MEL has images with and without the microscope in training, so there is one δ
     and the pooled / per_class modes coincide), then a class-balanced multinomial logistic regression refit on the
     training set. C ∈ {0.01, 0.1, 1, 10} chosen by the mean of clean and biased validation accuracy (section 3).
     Artifact label at training time only: microscope inserted (yes/no).
  b) DINOv2 ViT-B/14 @ 518 on lesion-masked images (p2_render.py images as their network sees them, lesion ROI from
     p2_lesion_rois.py; outside the ROI replaced by the mean colour, as wtss.ops.apply_roi_mask) + the same mask_cmc head.
  c) U-MtE (src/wtss/umte.py, ROI = lesion), label-free: subspace from the training images' masked vs generic-overlay
     features; class-balanced multinomial logistic regression on the erased features, C on validation as above.
  b and c share the masked DINOv2 features (U-MtE's masked view is arm b's view). `--stage features` (GPU) caches them;
  `--stage heads` (CPU) fits and scores.

Test (clean and biased), scored once: accuracy with an image-only bootstrap CI (one seed, A3), and the paired difference
to our Vanilla on the same images. Compared descriptively with the paper's Table 1 numbers.

    python scripts/replication/p2_arms.py --arm a
    python scripts/replication/p2_arms.py --arm bc --stage features     # GPU
    python scripts/replication/p2_arms.py --arm bc --stage heads        # CPU
Writes results/replication/pahde/arm_<arm>.{csv,md} and per-image predictions to $WTSS_DATA/replication/pahde/arms/.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
R = Path(f"{D}/replication/pahde")
OUT = ROOT / "results/replication/pahde"
CS = (0.01, 0.1, 1.0, 10.0)
SEED = 0
PAPER = {"Vanilla": (0.87, 0.28), "RRR": (0.84, 0.51), "RR-ClArC": (0.86, 0.76)}   # Table 1, ResNet-50, ISIC 2019


def boot(hit, n=10000, seed=20260918):
    """Image-only Poisson bootstrap of mean(hit) (hit: n_images or n_images x k, paired columns share the weights)."""
    rng = np.random.default_rng(seed)
    out = []
    for b0 in range(0, n, 500):
        w = rng.poisson(1.0, size=(min(500, n - b0), len(hit)))
        out.append((w @ hit) / np.maximum(w.sum(1, keepdims=True) if hit.ndim == 2 else w.sum(1), 1))
    return np.concatenate(out)


SETS = ("train_as_trained", "val_clean", "val_biased", "test_clean", "test_biased")


def select_and_score(name, Z, part, ytr):
    """Class-balanced multinomial LR, C by mean of clean + biased validation accuracy; test scored once."""
    sel, best = [], None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED).fit(Z["train"], ytr)
        vc = float((clf.predict(Z["val_clean"]) == part["val_clean"].label).mean())
        vb = float((clf.predict(Z["val_biased"]) == part["val_biased"].label).mean())
        sel.append({"C": C, "val_clean_acc": vc, "val_biased_acc": vb, "val_mean": (vc + vb) / 2, "converged": bool(clf.n_iter_.max() < 5000)})
        print(name, sel[-1], flush=True)
        if best is None or (vc + vb) / 2 > best[1]:
            best = (clf, (vc + vb) / 2, C)
    clf, _, C = best
    van = pd.read_csv(R / "scores/vanilla_seed0.csv.gz", usecols=["index", "split", "test_set", "label", "pred"])
    preds, rows = [], []
    for ts in ("clean", "biased"):
        p = part[f"test_{ts}"]
        v = van[(van.split == "test") & (van.test_set == ts)].set_index("index").loc[p["index"]]
        pr = clf.predict(Z[f"test_{ts}"])
        preds.append(pd.DataFrame({"index": p["index"], "image": p.image, "label": p.label, "method": name, "test_set": ts, "pred": pr}))
        hit = np.stack([(pr == p.label.to_numpy()), (v.pred.to_numpy() == p.label.to_numpy())], 1).astype(float)
        bs = boot(hit)
        lo, hi = np.quantile(bs[:, 0], [0.025, 0.975])
        dlo, dhi = np.quantile(bs[:, 0] - bs[:, 1], [0.025, 0.975])
        rows.append({"test": ts, "n": len(p), "arm_acc": hit[:, 0].mean(), "ci95_lo": lo, "ci95_hi": hi, "vanilla_acc": hit[:, 1].mean(),
                     "delta_vs_vanilla": hit[:, 0].mean() - hit[:, 1].mean(), "delta_ci95_lo": dlo, "delta_ci95_hi": dhi})
    (R / "arms").mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_csv(R / f"arms/preds_{name}.csv.gz", index=False)
    return pd.DataFrame(sel).assign(chosen=lambda d: d.C == C), pd.DataFrame(rows)


def write_report(key, title, sel, res):
    sel.to_csv(OUT / f"arm_{key}_selection.csv", index=False)
    res.to_csv(OUT / f"arm_{key}.csv", index=False)
    md = [f"# P2 — {title} (ISIC 2019, seed 0)", "",
          "One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not "
          "included. Δ vs Vanilla is paired on the same test images. Paper numbers (Table 1, ResNet-50, ISIC 2019) are "
          "descriptive.", "", "## Selection (validation: mean of clean and biased accuracy)", "",
          "| C | val clean | val biased | mean | chosen |", "|---|---|---|---|---|"]
    md += [f"| {r.C:g} | {r.val_clean_acc:.3f} | {r.val_biased_acc:.3f} | {r.val_mean:.3f} | {'**yes**' if r.chosen else ''} |" for _, r in sel.iterrows()]
    md += ["", "## Test (scored once)", "", "| test | n | arm [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] | paper Vanilla | paper RRR | paper RR-ClArC |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        k = 0 if r.test == "clean" else 1
        md.append(f"| {r.test} | {r.n} | {r.arm_acc:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {r.vanilla_acc:.3f} | "
                  f"{r.delta_vs_vanilla:+.3f} [{r.delta_ci95_lo:+.3f}, {r.delta_ci95_hi:+.3f}] | {PAPER['Vanilla'][k]:.2f} | "
                  f"{PAPER['RRR'][k]:.2f} | {PAPER['RR-ClArC'][k]:.2f} |")
    (OUT / f"arm_{key}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def bc_inputs():
    from PIL import Image
    z = np.load(R / "lesion_roi224.npz")
    roi = dict(zip(z["ids"].astype(str), np.unpackbits(z["bits"], axis=-1)[..., :224].astype(bool)))
    part = {k: pd.read_csv(R / f"render224/{k}.csv") for k in SETS}
    imgs = {k: np.load(R / f"render224/{k}.npy", mmap_mode="r") for k in SETS}
    for k in SETS:
        assert len(imgs[k]) == len(part[k]), k
    return part, imgs, roi, Image


def bc_features():
    """GPU: masked DINOv2-518 features of every set (arm b; U-MtE's X0) and the generic-overlay view of train (U-MtE X1)."""
    from wtss.umte import UMtE
    part, imgs, roi, Image = bc_inputs()
    um = UMtE("dino518", device="cuda", seed=SEED, batch_size=32)
    (R / "arms").mkdir(parents=True, exist_ok=True)
    jobs = [(k, False) for k in SETS] + [("train_as_trained", True)]
    for k, ins in jobs:
        f = R / f"arms/dino518_{'insert' if ins else 'masked'}_{k}.npz"
        if f.exists():
            continue
        ids = part[k].image.astype(str).tolist()
        X = []
        for s0 in range(0, len(ids), 1024):
            ims = [Image.fromarray(np.asarray(imgs[k][j])) for j in range(s0, min(s0 + 1024, len(ids)))]
            X.append(um.masked_features(ims, [roi[i] for i in ids[s0:s0 + 1024]], insert=ins, keys=ids[s0:s0 + 1024] if ins else None))
            print(k, "insert" if ins else "masked", s0 + len(ims), "/", len(ids), flush=True)
        np.savez(f, ids=np.array(ids, dtype="U"), X=np.concatenate(X))


def bc_heads():
    """CPU: arm b (mask_cmc on masked features) and arm c (U-MtE erased features)."""
    import candidates as CA
    from wtss.methods.insertion import fit_difference_subspace
    part = {k: pd.read_csv(R / f"render224/{k}.csv") for k in SETS}
    X = {k: np.load(R / f"arms/dino518_masked_{k}.npz")["X"] for k in SETS}
    X1 = np.load(R / "arms/dino518_insert_train_as_trained.npz")["X"]
    key = {"train_as_trained": "train"}
    P_ = {key.get(k, k): v for k, v in part.items()}
    ytr, atr = P_["train"].label.to_numpy(), P_["train"].artifact.to_numpy()
    Xk = {key.get(k, k): v for k, v in X.items()}
    P = CA.Projector(CA.cmc_directions(Xk["train"], ytr, atr, "pooled"))
    assert P.Q.shape[1] == 1, P.Q.shape
    sel, res = select_and_score("b_dino518_mask_cmc", {k: P(v) for k, v in Xk.items()}, P_, ytr)
    write_report("b", "arm b: DINOv2 ViT-B/14 @ 518 on lesion-masked images + mask_cmc", sel, res)
    er = fit_difference_subspace(Xk["train"], X1, energy=0.90, max_k=64, seed=SEED)
    sel, res = select_and_score("c_umte", {k: er(v) for k, v in Xk.items()}, P_, ytr)
    write_report("c", f"arm c: U-MtE (ROI = lesion, k = {er.k}), label-free", sel, res)


def arm_a():
    import candidates as CA
    F = np.load(R / "scores/vanilla_seed0_features.npz")
    s = pd.read_csv(R / "scores/vanilla_seed0.csv.gz", usecols=["index", "image", "split", "test_set", "label", "pred", "artifact"])
    part = {k: s[(s.split == sp) & (s.test_set == ts)].reset_index(drop=True)
            for k, (sp, ts) in {"train": ("train", "as_trained"), "val_clean": ("val", "clean"), "val_biased": ("val", "biased"),
                                "test_clean": ("test", "clean"), "test_biased": ("test", "biased")}.items()}
    X = {"train": F["train_as_trained"], **{k: F[k] for k in ("val_clean", "val_biased", "test_clean", "test_biased")}}
    for k in X:
        assert len(X[k]) == len(part[k]), k
    ytr, atr = part["train"].label.to_numpy(), part["train"].artifact.to_numpy()
    P = CA.Projector(CA.cmc_directions(X["train"], ytr, atr, "pooled"))
    assert P.Q.shape[1] == 1, P.Q.shape          # one δ (MEL); per_class gives the same span
    Z = {k: P(v) for k, v in X.items()}
    sel = []
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED).fit(Z["train"], ytr)
        vc = float((clf.predict(Z["val_clean"]) == part["val_clean"].label).mean())
        vb = float((clf.predict(Z["val_biased"]) == part["val_biased"].label).mean())
        sel.append({"C": C, "val_clean_acc": vc, "val_biased_acc": vb, "val_mean": (vc + vb) / 2, "converged": bool(clf.n_iter_.max() < 5000)})
        print(sel[-1], flush=True)
        if best is None or (vc + vb) / 2 > best[1]:
            best = (clf, (vc + vb) / 2, C)
    clf, _, C = best
    preds, rows = [], []
    for ts in ("clean", "biased"):
        p = part[f"test_{ts}"]
        pr = clf.predict(Z[f"test_{ts}"])
        preds.append(pd.DataFrame({"index": p["index"], "image": p.image, "label": p.label, "method": "a_mask_cmc_vanilla",
                                   "test_set": ts, "pred": pr}))
        hit = np.stack([(pr == p.label).to_numpy(), (p.pred == p.label).to_numpy()], 1).astype(float)   # ours, Vanilla
        bs = boot(hit)
        lo, hi = np.quantile(bs[:, 0], [0.025, 0.975])
        dlo, dhi = np.quantile(bs[:, 0] - bs[:, 1], [0.025, 0.975])
        rows.append({"test": ts, "n": len(p), "arm_a_acc": hit[:, 0].mean(), "ci95_lo": lo, "ci95_hi": hi,
                     "vanilla_acc": hit[:, 1].mean(), "delta_vs_vanilla": hit[:, 0].mean() - hit[:, 1].mean(),
                     "delta_ci95_lo": dlo, "delta_ci95_hi": dhi})
    (R / "arms").mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_csv(R / "arms/preds_a_mask_cmc_vanilla.csv.gz", index=False)
    return pd.DataFrame(sel).assign(chosen=lambda d: d.C == C), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "bc"], required=True)
    ap.add_argument("--stage", choices=["features", "heads"], default="heads")
    args = ap.parse_args()
    if args.arm == "bc":
        bc_features() if args.stage == "features" else bc_heads()
        return
    sel, res = arm_a()
    sel.to_csv(OUT / "arm_a_selection.csv", index=False)
    res.to_csv(OUT / "arm_a.csv", index=False)
    md = ["# P2 — arm a: mask_cmc on our Vanilla ResNet-50d (ISIC 2019, seed 0)", "",
          "One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not "
          "included. Δ vs Vanilla is paired on the same test images. Paper numbers (Table 1, ResNet-50, ISIC 2019) are "
          "descriptive; the head-to-head against RR-ClArC waits for the RR-ClArC decision (STATUS 2026-11-18).", "",
          "## Selection (validation: mean of clean and biased accuracy)", "", "| C | val clean | val biased | mean | chosen |", "|---|---|---|---|---|"]
    md += [f"| {r.C:g} | {r.val_clean_acc:.3f} | {r.val_biased_acc:.3f} | {r.val_mean:.3f} | {'**yes**' if r.chosen else ''} |" for _, r in sel.iterrows()]
    md += ["", "## Test (scored once)", "", "| test | n | arm a [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] | paper Vanilla | paper RRR | paper RR-ClArC |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        k = 0 if r.test == "clean" else 1
        md.append(f"| {r.test} | {r.n} | {r.arm_a_acc:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {r.vanilla_acc:.3f} | "
                  f"{r.delta_vs_vanilla:+.3f} [{r.delta_ci95_lo:+.3f}, {r.delta_ci95_hi:+.3f}] | {PAPER['Vanilla'][k]:.2f} | "
                  f"{PAPER['RRR'][k]:.2f} | {PAPER['RR-ClArC'][k]:.2f} |")
    (OUT / "arm_a.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
