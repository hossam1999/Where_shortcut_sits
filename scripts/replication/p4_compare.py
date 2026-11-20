"""P4 (Saab, MLHC 2022) second-tier comparison (A40; descriptive): our arms a, b, c, d vs our A31 Segmentation-ERM run of
their official code (`seg_pil_split1-3`), Bissoto ISIC 2018 trap sets 1-3, the same 772 test images; no claim against 0.727.
Registered P4 settings (A13 item 8): fit on the 1,822 training images, select on Bissoto's 259 validation images (inside
training, as A13 notes), score once on the 772 test images; flags with >= 3 fit images per artifact x diagnosis group.
  a  mask_cmc on the Seg-ERM encoder (scores as computed by p4_arm_a.py with P4_RUN=seg_pil, 2026-11-19)
  b  DINOv2 @ 518 lesion-masked (P8's features of the same A31 PIL cache), mask_cmc, C {0.01, 0.1, 1, 10}
  c  U-MtE (P8's insertion-view features of the same training images), C {0.01, 0.1, 1, 10}
  d  EfficientNet-B0 on lesion-masked 512-px images (A20 recipe via armd_train's P4 setting, one encoder per trap set),
     mask_cmc on the A21 grid
Statistics: 3-set mean robust AUROC, arm vs Seg-ERM, image-only bootstrap with one Poisson weight per image shared across
the trap sets and both methods, sets fixed (A32's estimator); Holm over the 4 arm comparisons; descriptive.
    venv-wtss: p4_compare.py --stage armd   (GPU)      p4_compare.py --stage compare   (CPU)
"""
import argparse, os, sys
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = Path(f"{D}/results_push/results/replication/saab/a40")
CS = (0.01, 0.1, 1.0, 10.0)


def setting(k):
    import armd_train as A

    class P4k(A.P4):
        SPLIT, KEY = k, f"p4_split{k}"
    return P4k()


def armd():
    import armd_train as A
    for k in (1, 2, 3):
        S = setting(k); out = Path(f"{D}/replication/armd/p4_split{k}"); out.mkdir(parents=True, exist_ok=True)
        if not (out / "TRAIN_DONE").exists():
            A.train(S, out, 12)
        if not (out / "features.npz").exists():
            A.features(S, out, 12)
        print("arm d p4 split", k, "done", flush=True)


def compare():
    import candidates as CA
    import armd_train as A
    from p4_arms_bc import rows, FLAGS
    from p8_arms import _select, boot_sets
    from wtss.methods.insertion import fit_difference_subspace
    from wtss.replication_stats import holm
    from sklearn.metrics import roc_auc_score
    zb = np.load(f"{D}/p8_arms/dino518_masked_all.npz"); bpos = {i: j for j, i in enumerate(zb["ids"].astype(str))}
    sa = pd.read_csv(ROOT / "results/replication/saab/scores/p4_arm_a_seg_pil.csv.gz", dtype={"image_id": str})
    per = {a: {} for a in ("a_segerm_mask_cmc", "b_dino518_mask_cmc", "c_umte", "d_effb0_mask_cmc")}; info = []
    for k in (1, 2, 3):
        r = rows(k); r = {"fit": r["train"], "val": r["val"], "test": r["test"]}
        ys = {s: r[s].target.to_numpy() for s in r}; tid = r["test"].id.tolist()
        used = [f for f in FLAGS if (pd.crosstab(r["fit"][f], r["fit"].target).reindex(index=[0, 1], columns=[0, 1], fill_value=0) >= 3).all().all()]
        Afl = r["fit"][used].to_numpy().astype(int)
        seg = pd.read_csv(f"{D}/replication/saab/runs/seg_pil_split{k}/preds_test.csv"); seg = seg.set_index(seg.id.astype(str)).loc[tid, "score"].to_numpy()
        Xb = {s: zb["X"][[bpos[i] for i in r[s].id]] for s in r}
        sc, inf, _ = _select(Xb, ys, Afl, CA.CMC_MODES, CS); per["b_dino518_mask_cmc"][k] = (tid, ys["test"], sc, seg); info.append({"split": k, "arm": "b", **inf})
        ins = np.load(f"{D}/p8_arms/dino518_insert_train{k}.npz"); ipos = {i: j for j, i in enumerate(ins["ids"].astype(str))}
        er = fit_difference_subspace(Xb["fit"], ins["X"][[ipos[i] for i in r["fit"].id]], energy=0.90, max_k=64, seed=0)
        sc, inf, _ = _select({s: er(v) for s, v in Xb.items()}, ys, Afl, (None,), CS); per["c_umte"][k] = (tid, ys["test"], sc, seg); info.append({"split": k, "arm": "c", **inf})
        F = np.load(f"{D}/replication/armd/p4_split{k}/features.npz")
        sc, inf, _ = _select({"fit": F["train"], "val": F["val"], "test": F["test"]}, ys, Afl, CA.CMC_MODES, A.CS)
        per["d_effb0_mask_cmc"][k] = (tid, ys["test"], sc, seg); info.append({"split": k, "arm": "d", **inf})
        a = sa[sa.split == k].set_index("image_id").loc[tid]; per["a_segerm_mask_cmc"][k] = (tid, ys["test"], a.score.to_numpy(), seg)
    out = [{"arm": arm, **boot_sets(p)} for arm, p in per.items()]
    o = pd.DataFrame(out); o["p_holm"] = holm(o.p_boot_two_sided.tolist())
    for arm, p in per.items():
        for k, (_, y, s, g) in p.items():
            o.loc[o.arm == arm, f"set{k}_arm"] = roc_auc_score(y, s); o.loc[o.arm == arm, f"set{k}_segerm"] = roc_auc_score(y, g)
    OUT.mkdir(parents=True, exist_ok=True); o.to_csv(OUT / "comparison.csv", index=False); pd.DataFrame(info).to_csv(OUT / "selection.csv", index=False)
    md = ["# P4 second-tier comparison (A40; DESCRIPTIVE) — our arms vs our A31 Seg-ERM run of Saab's official code — machine 2", "",
          "P4 is not reproduced (Seg-ERM 0.647 vs 0.727; A38). Robust AUROC on the 772-image trap tests, trap sets 1–3; image-only "
          "bootstrap with one Poisson weight per image shared across sets and methods, sets fixed; Holm over 4. No claim against 0.727.", "",
          "| arm | set 1 | set 2 | set 3 | 3-set mean | Seg-ERM mean | Δ [95 % CI] | p Holm |", "|---|---|---|---|---|---|---|---|"]
    for _, x in o.iterrows():
        md.append(f"| {x.arm} | {x.set1_arm:.3f} | {x.set2_arm:.3f} | {x.set3_arm:.3f} | {x.ours_mean:.3f} | {x.theirs_mean:.3f} | "
                  f"{x.delta:+.3f} [{x.ci95_lo:+.3f}, {x.ci95_hi:+.3f}] | {x.p_holm:.4f} |")
    (OUT / "comparison.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["armd", "compare"], required=True)
    {"armd": armd, "compare": compare}[ap.parse_args().stage]()
