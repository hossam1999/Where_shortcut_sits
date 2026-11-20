"""P8 (DISC, ICML 2023): our arms a-d vs our reproduced DISC on the Bissoto ISIC 2018 trap sets 1-5 (A32, A35, A37).

Splits per trap set k (A35), from the trap-set CSVs (= DISC's `isic/trap-sets`, byte-identical to Bissoto's):
  primary   (DISC's protocol): fit = train CSV (1,822); val / test = DISC's own 154 / 618 rows of the test CSV (their
            ISICDataset's labelling of train_test_split(..., test_size=0.8, random_state=0); see split(), A35 correction);
  secondary (A35 addendum):    fit = train CSV minus Bissoto's val CSV (1,563); val = Bissoto's val CSV (259); same test.
Arms (registered P4 settings, re-run unchanged; lesion ROI = ISIC 2018 Task 1 masks, outside = mean colour, A31 PIL cache):
  a  mask_cmc on the reproduced DISC ResNet-50 of trap set k (penultimate 2,048-d of lesion-masked images through DISC's
     eval transform), C grid {0.01, 0.1, 1, 10};
  b  DINOv2 ViT-B/14 @ 518, lesion-masked, mask_cmc, C grid {0.01, 0.1, 1, 10};
  c  U-MtE (label-free difference subspace of the insertion view), C grid {0.01, 0.1, 1, 10};
  d  EfficientNet-B0 trained on lesion-masked 512-px images (A20 recipe, early stopping on val AUROC), mask_cmc, A21 grid;
     one encoder per (trap set, rule).
Head: flags with >= 3 fit images in each artifact x diagnosis group (A13 item 8, on the fit set of the rule); mask_cmc mode
and C by validation AUROC; test scored once.
Comparison (A32): 5-set mean test AUROC, arm vs reproduced DISC, paired on the same test images; 10,000 replicates, one
Poisson(1) weight per image id shared across trap sets and both methods, trap sets fixed; per-set AUROC recomputed and
averaged. "Beaten" = higher mean, 95 % CI above 0, Holm-adjusted p < 0.05; Holm family = the 4 primary-rule comparisons.
Secondary rule: same statistics, reported beside, outside the Holm family.

Stages (each skips finished work):
    venv-wtss  p8_arms.py --stage features_bc            GPU: DINOv2 masked (all 2,594 images) + U-MtE insert (fit images)
    envs/disc  p8_arms.py --stage features_a             GPU: DISC/ERM per-image test scores (their cached rows; must
                                                         reproduce the logged test AUROC) + arm-a features
    venv-wtss  p8_arms.py --stage armd [--set k --rule r] GPU: arm-d encoders + features (armd_train functions)
    venv-wtss  p8_arms.py --stage heads                  CPU: heads, both rules, per-image test scores
    venv-wtss  p8_arms.py --stage compare                CPU: bootstrap, Holm, tables
Outputs: $WTSS_DATA/p8_arms/ (features, encoders) and $WTSS_DATA/results_push/results/replication/disc/arms/ (pushed).
"""
import argparse, glob, json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
TS = f"{D}/disc/isic/trap-sets"
W = Path(f"{D}/p8_arms")
RES = Path(f"{D}/results_push/results/replication/disc/arms")
SETS, RULES = (1, 2, 3, 4, 5), ("primary", "secondary")
FLAGS = ("dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches", "vasc")
CS = (0.01, 0.1, 1.0, 10.0)
MEAN_RGB = (124, 116, 104)   # wtss.utils.MEAN_RGB (checked in heads)


def _read(name, k):
    d = pd.read_csv(f"{TS}/isic_annotated_{name}{k}.csv").rename(columns={"label": "target"})
    d["id"] = d.image.str.split(".").str[0]; d["image_id"] = d["id"]
    return d.reset_index(drop=True)


def split(k, rule):
    from sklearn.model_selection import train_test_split
    tr, te = _read("train", k), _read("test", k)
    iv, it = train_test_split(np.arange(len(te)), test_size=0.8, random_state=0)
    # DISC's ISICDataset lists the rows after train as [iv, it] but labels them [test (len it), val (len iv)]
    # (isic_dataset.py:77-89): their test = iv + it[:len(it) - len(iv)], their val = the last len(iv) rows of it (A35 correction)
    order = np.concatenate([iv, it])
    test = te.iloc[order[:len(it)]].reset_index(drop=True)
    if rule == "primary":
        fit, val = tr, te.iloc[order[len(it):]].reset_index(drop=True)
    else:
        val = _read("val", k); fit = tr[~tr.id.isin(set(val.id))].reset_index(drop=True)
    assert not set(fit.id) & set(test.id) and not set(val.id) & set(test.id) and not set(fit.id) & set(val.id)
    return {"fit": fit, "val": val, "test": test}


def used_flags(fit):
    return [f for f in FLAGS if (pd.crosstab(fit[f], fit.target).reindex(index=[0, 1], columns=[0, 1], fill_value=0) >= 3).all().all()]


def all_ids():
    ids = sorted({i for k in SETS for n in ("train", "test") for i in _read(n, k).id})
    # p4_train.cache() rebuilds the shared 512-px cache when its id list differs: never let that happen here
    assert ids == Path(f"{D}/replication/saab/cache512_pil/ids.txt").read_text().split(), "id list differs from the A31 cache"
    return ids


# ------------------------------------------------------------------------------------------------------ features
def features_bc():
    from PIL import Image
    import p4_train as T
    from wtss.umte import UMtE
    ids = all_ids(); img, msk, cids = T.cache(ids); assert cids == ids
    W.mkdir(parents=True, exist_ok=True)
    um = UMtE("dino518", device="cuda", seed=0, batch_size=16)
    f = W / "dino518_masked_all.npz"
    if not f.exists():
        X = []
        for s0 in range(0, len(ids), 256):
            js = range(s0, min(s0 + 256, len(ids)))
            X.append(um.masked_features([Image.fromarray(np.asarray(img[j])) for j in js], [np.asarray(msk[j]) > 0 for j in js]))
            print("masked", js[-1] + 1, "/", len(ids), flush=True)
        np.savez(f, ids=np.array(ids, dtype="U"), X=np.concatenate(X))
    pos = {i: j for j, i in enumerate(ids)}
    for k in SETS:   # U-MtE's insertion view of trap set k's training images (the secondary fit set is a subset)
        g = W / f"dino518_insert_train{k}.npz"
        if g.exists():
            continue
        tid = _read("train", k).id.tolist(); X = []
        for s0 in range(0, len(tid), 256):
            js = [pos[i] for i in tid[s0:s0 + 256]]
            X.append(um.masked_features([Image.fromarray(np.asarray(img[j])) for j in js], [np.asarray(msk[j]) > 0 for j in js],
                                        insert=True, keys=tid[s0:s0 + 256]))
        np.savez(g, ids=np.array(tid, dtype="U"), X=np.concatenate(X)); print("insert train", k, flush=True)


def features_a():
    """DISC env. Per trap set: (1) DISC's and ERM's per-image test scores from their own cache rows (the tensors their
    run_expt.py scored; cache verified row by row, A37), softmax[:, 1], asserted to reproduce the logged test AUROC of the
    best-validation epoch; (2) arm-a features: penultimate 2,048-d of DISC's best model on lesion-masked images through
    DISC's eval transform (Resize 224, ColorJitter(hue=0.2), ToTensor, ImageNet normalisation; torch seed 0)."""
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/third_party/disc")
    from disc.dataset.transform import get_transform_ISIC
    c = Path(f"{D}/replication/saab/cache512_pil"); cids = c.joinpath("ids.txt").read_text().split()
    img, msk = np.load(c / "img.npy", mmap_mode="r"), np.load(c / "mask.npy", mmap_mode="r"); cpos = {i: j for j, i in enumerate(cids)}
    tf = get_transform_ISIC("resnet50", train=False, augment_data=False)
    W.mkdir(parents=True, exist_ok=True)
    for k in SETS:
        out = W / f"disc_set{k}.npz"
        if out.exists():
            continue
        sp = split(k, "primary"); n_tr, n_va, n_te = len(sp["fit"]), len(sp["val"]), len(sp["test"])
        cache = np.load(f"{D}/disc/isic_set{k}/features/feature_mat.npy", mmap_mode="r")
        assert len(cache) == n_tr + n_va + len(sp["test"])
        res = {}
        for m in ("ERM", "DISC"):
            d = glob.glob(f"{D}/disc/output/ISIC/{m}/*-trapset_id={k}"); assert len(d) == 1
            model = torch.load(f"{d[0]}/best_model.pth", map_location="cuda", weights_only=False).eval()
            p, F = [], []
            h = model.fc.register_forward_hook(lambda mod, i, o: F.append(i[0].float().cpu()))
            with torch.no_grad():
                for s0 in range(n_tr, n_tr + n_te, 128):   # their test rows follow train directly (see split())
                    x = torch.from_numpy(np.asarray(cache[s0:min(s0 + 128, n_tr + n_te)])).float().cuda()
                    p.append(torch.softmax(model(x), 1)[:, 1].cpu())
            h.remove(); s = torch.cat(p).numpy()
            v, t = pd.read_csv(f"{d[0]}/val.csv"), pd.read_csv(f"{d[0]}/test.csv"); logged = float(t.roc_auc.iloc[int(np.argmax(v.roc_auc.values))])
            auc = roc_auc_score(sp["test"].target, s)
            assert abs(auc - logged) < 1e-3, (m, k, auc, logged)   # their eval during training vs this pass: |diff| ~1e-4
            print(f"set {k} {m}: test AUROC from their cache rows {auc:.5f}, logged {logged:.5f}", flush=True)
            res[f"{m}_test_score"], res[f"{m}_logged_auroc"] = s, logged
            if m == "DISC":   # arm a: lesion-masked images of every id the two rules use
                ids = sorted(set(_read("train", k).id) | set(_read("test", k).id)); torch.manual_seed(0); X = []
                h = model.fc.register_forward_hook(lambda mod, i, o: F.append(i[0].float().cpu()))
                with torch.no_grad():
                    for s0 in range(0, len(ids), 64):
                        xs = []
                        for i in ids[s0:s0 + 64]:
                            a = np.asarray(img[cpos[i]]).copy(); a[np.asarray(msk[cpos[i]]) == 0] = np.asarray(MEAN_RGB, np.uint8)
                            xs.append(tf(Image.fromarray(a)))
                        F.clear(); model(torch.stack(xs).cuda()); X.append(F[0].numpy())
                h.remove(); res["arm_a_ids"], res["arm_a_X"] = np.array(ids, dtype="U"), np.concatenate(X)
            del model; torch.cuda.empty_cache()
        np.savez(out, test_ids=sp["test"].id.to_numpy().astype("U"), **res)
        print(f"set {k}: ERM {res['ERM_logged_auroc']:.4f} / DISC {res['DISC_logged_auroc']:.4f} reproduced from their cache rows; "
              f"arm-a features {res['arm_a_X'].shape}", flush=True)


# ------------------------------------------------------------------------------------------------------ arm d
class P8Setting:
    """armd_train setting for trap set k under one A35 rule (same images, masking and augmentation as P4's arm d)."""
    n_out = 2
    sets = ("fit", "val", "test")
    train_set, val_sets = "fit", ("val",)

    def __init__(self, k, rule):
        import p4_train as P4T
        self.T, self.KEY = P4T, f"p8_set{k}_{rule}"
        self.part = split(k, rule)
        self.img, self.msk, ids = P4T.cache(all_ids())
        self.pos = {i: j for j, i in enumerate(ids)}
        self.used = used_flags(self.part["fit"])

    def dataset(self, k, train):
        import armd_train as A
        return A.P4Masked(self.img, self.msk, self.part[k], self.pos, self.T.augmenter() if train else None)

    def val_metric(self, preds):
        from sklearn.metrics import roc_auc_score
        y, p = preds["val"]
        return float(roc_auc_score(y, p[:, 1] - p[:, 0]))

    def labels(self, k):
        d = self.part[k]
        return d.target.to_numpy().astype(int), d[self.used].to_numpy().astype(int)


def armd(sets, rules, workers):
    import armd_train as A
    for k in sets:
        for r in rules:
            S = P8Setting(k, r); out = W / "armd" / S.KEY; out.mkdir(parents=True, exist_ok=True)
            if not (out / "TRAIN_DONE").exists():
                A.train(S, out, workers)
            if not (out / "features.npz").exists():
                A.features(S, out, workers)
            print("arm d", S.KEY, "done", flush=True)


# ------------------------------------------------------------------------------------------------------ heads
def _select(Xs, ys, A, modes, grid):
    """mask_cmc directions of every used flag (union, orthonormalised), class-balanced LR; (mode, C) by val AUROC (first max)."""
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    best, sel = None, []
    for mode in modes:
        if mode is None:
            Z = Xs
        else:
            Q = np.concatenate([CA.cmc_directions(Xs["fit"], ys["fit"], A[:, j], mode) for j in range(A.shape[1])], 1)
            q, rr = np.linalg.qr(Q); Q = q[:, np.abs(np.diag(rr)) > 1e-10 * max(1.0, np.abs(rr).max())]
            P = CA.Projector(Q); Z = {s: P(v) for s, v in Xs.items()}
        for C in grid:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=0).fit(Z["fit"], ys["fit"])
            v = float(roc_auc_score(ys["val"], clf.decision_function(Z["val"])))
            sel.append({"mode": mode or "none", "C": C, "val_auroc": v})
            if best is None or v > best[1]:
                best = (clf, v, mode or "none", C, Z)
    clf, v, mode, C, Z = best
    return clf.decision_function(Z["test"]), {"mode": mode, "C": C, "val_auroc": v,
                                              "C_edge": "lowest" if C == grid[0] else "highest" if C == grid[-1] else ""}, sel


def heads():
    import candidates as CA
    from sklearn.metrics import roc_auc_score
    from wtss.utils import MEAN_RGB as M
    from wtss.methods.insertion import fit_difference_subspace
    import armd_train as A
    assert tuple(M) == MEAN_RGB
    zb = np.load(W / "dino518_masked_all.npz"); bpos = {i: j for j, i in enumerate(zb["ids"].astype(str))}
    rows, scores, sels = [], [], []
    for k in SETS:
        za = np.load(W / f"disc_set{k}.npz"); apos = {i: j for j, i in enumerate(za["arm_a_ids"].astype(str))}
        ins = np.load(W / f"dino518_insert_train{k}.npz"); ipos = {i: j for j, i in enumerate(ins["ids"].astype(str))}
        for r in RULES:
            sp = split(k, r); used = used_flags(sp["fit"]); Afl = sp["fit"][used].to_numpy().astype(int)
            ys = {s: sp[s].target.to_numpy() for s in sp}
            assert (sp["test"].id.to_numpy() == za["test_ids"].astype(str)).all()
            arms = {}
            Xb = {s: zb["X"][[bpos[i] for i in sp[s].id]] for s in sp}
            arms["b_dino518_mask_cmc"] = _select(Xb, ys, Afl, CA.CMC_MODES, CS)
            Xa = {s: za["arm_a_X"][[apos[i] for i in sp[s].id]] for s in sp}
            arms["a_disc_mask_cmc"] = _select(Xa, ys, Afl, CA.CMC_MODES, CS)
            er = fit_difference_subspace(Xb["fit"], ins["X"][[ipos[i] for i in sp["fit"].id]], energy=0.90, max_k=64, seed=0)
            arms["c_umte"] = _select({s: er(v) for s, v in Xb.items()}, ys, Afl, (None,), CS)
            fd = W / "armd" / f"p8_set{k}_{r}" / "features.npz"
            if fd.exists():
                Fd = np.load(fd); arms["d_effb0_mask_cmc"] = _select({s: Fd[s] for s in sp}, ys, Afl, CA.CMC_MODES, A.CS)
            for name, (sc, info, sel) in arms.items():
                rows.append({"trapset": k, "rule": r, "arm": name, "n_fit": len(sp["fit"]), "n_val": len(sp["val"]),
                             "test_auroc": float(roc_auc_score(ys["test"], sc)), "flags": ",".join(used), **info})
                scores.append(pd.DataFrame({"trapset": k, "rule": r, "method": name, "image_id": sp["test"].id, "label": ys["test"], "score": sc}))
                sels.append(pd.DataFrame(sel).assign(trapset=k, rule=r, arm=name))
        for m in ("DISC", "ERM"):
            scores.append(pd.DataFrame({"trapset": k, "rule": "their", "method": m, "image_id": za["test_ids"].astype(str),
                                        "label": split(k, "primary")["test"].target.to_numpy(), "score": za[f"{m}_test_score"]}))
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "arms_per_set.csv", index=False)
    pd.concat(scores).to_csv(RES / "scores_test.csv.gz", index=False)
    pd.concat(sels).to_csv(RES / "selection.csv", index=False)
    print(pd.DataFrame(rows)[["trapset", "rule", "arm", "mode", "C", "val_auroc"]].to_string())


# ------------------------------------------------------------------------------------------------------ compare
def boot_sets(per_set, n_boot=10000, seed=20260918, chunk=250):
    """A32: per_set = {k: (ids, y, s_arm, s_ref)}; one Poisson(1) weight per image id shared across trap sets and both
    methods; trap sets fixed; statistic = mean over sets of AUROC(arm) - AUROC(ref) on the weighted images."""
    from wtss.replication_stats import _metric
    from wtss.stats import _ci, boot_p
    rng = np.random.default_rng(seed)
    ids = sorted({str(i) for v in per_set.values() for i in v[0]}); pos = {i: j for j, i in enumerate(ids)}
    prep, pa, pb = {}, [], []
    for k, (im, y, a, b) in sorted(per_set.items()):
        fa, xa = _metric("auroc", y, a); fb, xb = _metric("auroc", y, b)
        prep[k] = (np.array([pos[str(i)] for i in im]), fa, fb); pa.append(xa); pb.append(xb)
    out = []
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0); Wt = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        out.append(np.mean([fa(Wt[:, idx]) - fb(Wt[:, idx]) for idx, fa, fb in prep.values()], 0))
    arr = np.concatenate(out); arr = arr[np.isfinite(arr)]; lo, hi = _ci(arr)
    return {"ours_mean": float(np.mean(pa)), "theirs_mean": float(np.mean(pb)), "delta": float(np.mean(pa) - np.mean(pb)),
            "ci95_lo": lo, "ci95_hi": hi, "p_boot_two_sided": boot_p(arr), "n_boot_valid": int(len(arr))}


def compare():
    from wtss.replication_stats import holm, beaten
    s = pd.read_csv(RES / "scores_test.csv.gz", dtype={"image_id": str})
    ref = s[s.method == "DISC"].set_index(["trapset", "image_id"]).score
    out = []
    for r in RULES:
        for arm in sorted(s[(s.rule == r)].method.unique()):
            per = {}
            for k in SETS:
                x = s[(s.rule == r) & (s.method == arm) & (s.trapset == k)]
                per[k] = (x.image_id.tolist(), x.label.to_numpy(), x.score.to_numpy(), ref.loc[k].loc[x.image_id].to_numpy())
            out.append({"rule": r, "arm": arm, "primary_arm": arm[0] in "bd", **boot_sets(per)})
    o = pd.DataFrame(out)
    m = o.rule == "primary"; o.loc[m, "p_holm"] = holm(o.loc[m, "p_boot_two_sided"].tolist())
    o["meets_beaten_rule"] = [bool(beaten(x, x.p_holm)) if x.rule == "primary" else None for _, x in o.iterrows()]
    g = json.load(open(RES.parent / "gate.json"))
    o["gate_passed"] = g["gate_passed"]   # A32: verdicts only if the gate passes; otherwise descriptive (tier 2)
    o.to_csv(RES / "comparison.csv", index=False)
    e = s[s.method == "ERM"]
    status = ("**Gate PASSED: verdicts apply.**" if g["gate_passed"] else
              f"**Gate FAILED (ERM {g['ERM']['mean']:.3f} vs 0.364, DISC {g['DISC']['mean']:.3f} vs 0.551; A37 reading): P8 is "
              "DESCRIPTIVE.** The comparison is against our run of DISC's official code under identical conditions (reporting "
              "tier 2), not against the published 0.551; the last column says whether the registered rule's conditions hold, "
              "not a verdict.")
    md = ["# P8: our arms vs our reproduced DISC (A32, A35, A37) — machine 2", "", status, "",
          "5-set mean test ROC-AUC (618 test images per trap set, DISC's split); paired image-only bootstrap (10,000 "
          "replicates, one Poisson weight per image shared across trap sets and methods, trap sets fixed). Primary rule = "
          "DISC's protocol (selection on 154 test-distribution validation images, as DISC and ERM); secondary = fit on "
          "1,563 / select on Bissoto's 259 held-out validation images (no test-distribution data). Holm family = the 4 "
          "primary comparisons; arms b and d primary, a and c secondary.", "",
          "| rule | arm | arm mean | DISC mean | Δ [95 % CI] | p | p Holm | rule conditions met |", "|---|---|---|---|---|---|---|---|"]
    for _, x in o.iterrows():
        md.append(f"| {x.rule} | {x.arm} | {x.ours_mean:.3f} | {x.theirs_mean:.3f} | {x.delta:+.3f} [{x.ci95_lo:+.3f}, "
                  f"{x.ci95_hi:+.3f}] | {x.p_boot_two_sided:.4f} | " + (f"{x.p_holm:.4f} | {'yes' if x.meets_beaten_rule else 'no'} |"
                                                                       if x.rule == "primary" else "— | — |"))
    md += ["", "Published-number row (descriptive): **CBDebug (Enouen & Galhotra, ICLR 2026) 0.580 ± 0.116** on the same Bissoto "
           "trap sets — not reproduced: no ISIC code released (published numbers come from a protocol we could not check; A37)."]
    md += ["", "Per trap set: `arms_per_set.csv`; per-image test scores (arms, DISC, ERM): `scores_test.csv.gz`; selection grids: "
           "`selection.csv`. ERM 5-set mean (same images): " +
           f"{np.mean([__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(g.label, g.score) for _, g in e.groupby('trapset')]):.3f}."]
    (RES / "comparison.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features_bc", "features_a", "armd", "heads", "compare"], required=True)
    ap.add_argument("--set", type=int, nargs="*", default=list(SETS)); ap.add_argument("--rule", nargs="*", default=list(RULES))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"features_bc": features_bc, "features_a": features_a, "heads": heads, "compare": compare,
     "armd": lambda: armd(a.set, a.rule, a.workers)}[a.stage]()
