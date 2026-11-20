"""P6 (EPVT, MICCAI 2023; A39): our arms a, b, c, d vs our EPVT run (5 seeds) on Bissoto's ISIC 2019 trap sets, bias 1 and
0.9, split 1. Fit on the EPVT training images of the trap set (`train_bias_<b>_1.csv` = their grouped folders), select on
`val_bias_<b>_1`, score once on `test_bias_<b>_1`; binary melanoma, AUROC.
Lesion ROI (A39, as P2): HAM10000 manual mask, else ISIC 2018 Task 1, else the lesion U-Net (rebuilt on machine 2); outside =
mean colour (wtss.utils.MEAN_RGB).
  a  mask_cmc on our EPVT run: 768-d input of EPVT's classifier in its own predict() (prompted pass) on lesion-masked images
     through their eval transform (Resize (224, 224), ImageNet normalisation); one head per EPVT seed; C {0.01, 0.1, 1, 10}
  b  DINOv2 ViT-B/14 @ 518, lesion-masked (wtss.umte masked view of the original image), mask_cmc, C {0.01, 0.1, 1, 10}
  c  U-MtE (label-free difference subspace of the insertion view of the fit images), C {0.01, 0.1, 1, 10}
  d  EfficientNet-B0 trained on lesion-masked images (A20 recipe via armd_train: Adam 1e-4, batch 64, <= 50 epochs, early
     stopping on validation AUROC, patience 5, seed 1234; EPVT's training augmentation on the 256-px masked image), mask_cmc
     on the A21 grid; one encoder per bias level.
Flags (heads): Bissoto's flags with >= 3 fit images in each artifact x diagnosis group (A13 item 8); mode and C by
validation AUROC (first maximum). Comparison: crossed seed x image bootstrap vs EPVT's 5 seeds (arm a seed by seed; b, c, d
paired with every seed); Holm over 8 (4 arms x 2 bias levels); section 2 "beaten" rule.

Stages:   cache (CPU) -> features_a (EPVT env, GPU) -> features_bc (venv-wtss, GPU) -> armd (GPU) -> heads (CPU) -> compare
    --bias 1 [0.9]   (each stage skips finished work)
"""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
TS = "/workspace/third_party/bissoto_ag/trap_sets"
IMG = f"{D}/epvt/isic2019_plain"
W = Path(f"{D}/p6_arms")
RES = Path(f"{D}/results_push/results/replication/epvt/arms")
FLAGS = ("dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches", "vasc")
CS = (0.01, 0.1, 1.0, 10.0)
MEAN_RGB = (124, 116, 104)
Image.MAX_IMAGE_PIXELS = None


def bstr(b):
    return "1" if float(b) == 1 else str(b)


def split(b):
    p = {s: pd.read_csv(f"{TS}/{s}_bias_{bstr(b)}_1.csv") for s in ("train", "val", "test")}
    return {"fit": p["train"], "val": p["val"], "test": p["test"]}


def used_flags(fit):
    return [f for f in FLAGS if (pd.crosstab(fit[f], fit.label).reindex(index=[0, 1], columns=[0, 1], fill_value=0) >= 3).all().all()]


def mask_source():
    from wtss import paths
    ham = {p.name.split("_segmentation")[0]: p for p in Path(f"{D}/ham_seg").rglob("*_segmentation.png")}
    i18 = {p.name.split("_segmentation")[0]: p for p in Path(f"{D}/isic2018/ISIC2018_Task1_Training_GroundTruth").glob("*_segmentation.png")}
    un = Path(f"{D}/isic2019/unet_masks")
    # ISIC names 2,074 originals `*_downsampled`; the U-Net writes its masks under the original file names
    unp = lambda i: un / f"{i}_unet.png" if (un / f"{i}_unet.png").exists() else un / f"{i}_downsampled_unet.png"
    return lambda i: ("ham", ham[i]) if i in ham else ("isic2018", i18[i]) if i in i18 else ("unet", unp(i))


def load(i, size, src):
    """original image and its lesion mask, both resized to size x size (squashed, as EPVT's Resize((224, 224)))"""
    with Image.open(f"{IMG}/{i}.jpg") as im:
        im.draft("RGB", (2 * size, 2 * size)); x = im.convert("RGB").resize((size, size), Image.BILINEAR)
    s, p = src(i)
    with Image.open(p) as m:
        r = np.asarray(m.convert("L").resize((size, size), Image.NEAREST)) >= 128
    return np.asarray(x, np.uint8), r, s


def all_ids(biases):
    return sorted({i for b in biases for v in split(b).values() for i in v.image})


# ---------------------------------------------------------------------------------------------------------- cache
def cache(biases):
    src = mask_source(); W.mkdir(parents=True, exist_ok=True)
    ids = all_ids(biases)
    for size in (224, 256):
        d = W / f"cache{size}"
        if (d / "ids.txt").exists() and (d / "ids.txt").read_text().split() == ids:
            continue
        d.mkdir(exist_ok=True)
        I = np.lib.format.open_memmap(d / "img.npy", "w+", np.uint8, (len(ids), size, size, 3))
        M = np.lib.format.open_memmap(d / "mask.npy", "w+", np.uint8, (len(ids), size, size))
        srcs = []
        with ThreadPoolExecutor(24) as ex:
            for k, (x, r, s) in enumerate(ex.map(lambda i: load(i, size, src), ids)):
                I[k], M[k] = x, r; srcs.append(s)
        I.flush(); M.flush()
        (d / "ids.txt").write_text("\n".join(ids))
        t = pd.DataFrame({"image": ids, "roi_source": srcs, "roi_frac": np.asarray(M).reshape(len(ids), -1).mean(1)})
        t.to_csv(d / "roi_sources.csv", index=False)
        RES.mkdir(parents=True, exist_ok=True)
        t.groupby("roi_source").agg(n=("roi_frac", "size"), mean_roi_frac=("roi_frac", "mean"),
                                    empty=("roi_frac", lambda v: int((v == 0).sum()))).to_csv(RES / "roi_sources.csv")
        print("cache", size, len(ids), flush=True)


def cached(size):
    d = W / f"cache{size}"
    ids = (d / "ids.txt").read_text().split()
    return np.load(d / "img.npy", mmap_mode="r"), np.load(d / "mask.npy", mmap_mode="r"), {i: j for j, i in enumerate(ids)}


def masked(x, r):
    x = np.asarray(x).copy(); x[~np.asarray(r, bool)] = np.asarray(MEAN_RGB, np.uint8); return x


# ------------------------------------------------------------------------------------------------------ arm a (EPVT env)
def features_a(biases):
    import torch
    sys.path.insert(0, "/workspace/third_party/epvt")
    from domainbed import algorithms
    I, M, pos = cached(224)
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for b in biases:
        sp = split(b)
        for s in range(5):
            out = W / f"a_bias{bstr(b)}_seed{s}.npz"
            if out.exists():
                continue
            ck = f"{D}/epvt/runs/epvt_a36_bias{bstr(b)}_1_seed{s}/epvt.pkl"
            c = torch.load(ck, map_location="cpu", weights_only=False)
            alg = algorithms.get_algorithm_class(c["args"]["algorithm"])(c["model_input_shape"], c["model_num_classes"], 5, c["model_hparams"])
            alg.load_state_dict(c["model_dict"]); alg.cuda().eval()
            F = []; h = alg.classifier.register_forward_hook(lambda m, i, o: F.append(i[0].float().cpu()))
            res = {}
            with torch.no_grad():
                for name, df in sp.items():
                    X = []
                    for s0 in range(0, len(df), 128):
                        js = [pos[i] for i in df.image.iloc[s0:s0 + 128]]
                        x = torch.from_numpy(np.stack([masked(I[j], M[j]) for j in js])).permute(0, 3, 1, 2).float() / 255
                        F.clear(); alg.predict(((x - mean) / std).cuda()); X.append(F[-1].numpy())   # last call = prompted pass
                    res[f"X_{name}"] = np.concatenate(X)
            h.remove(); np.savez(out, **res); print("arm a", b, s, {k: v.shape for k, v in res.items()}, flush=True)
            del alg; torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------ arms b, c
def features_bc(biases):
    from wtss.umte import UMtE
    src = mask_source(); W.mkdir(parents=True, exist_ok=True)
    um = UMtE("dino518", device="cuda", seed=0, batch_size=16)
    f = W / "dino518_masked_all.npz"
    ids = all_ids(biases)
    if not (f.exists() and list(np.load(f)["ids"]) == ids):
        X = []
        with ThreadPoolExecutor(16) as ex:
            for s0 in range(0, len(ids), 256):
                L = list(ex.map(lambda i: load(i, 518, src), ids[s0:s0 + 256]))
                X.append(um.masked_features([Image.fromarray(x) for x, _, _ in L], [r for _, r, _ in L])); print("masked", s0 + len(L), "/", len(ids), flush=True)
        np.savez(f, ids=np.array(ids, dtype="U"), X=np.concatenate(X))
    for b in biases:
        g = W / f"dino518_insert_fit_bias{bstr(b)}.npz"
        if g.exists():
            continue
        tid = split(b)["fit"].image.tolist(); X = []
        with ThreadPoolExecutor(16) as ex:
            for s0 in range(0, len(tid), 256):
                L = list(ex.map(lambda i: load(i, 518, src), tid[s0:s0 + 256]))
                X.append(um.masked_features([Image.fromarray(x) for x, _, _ in L], [r for _, r, _ in L], insert=True, keys=tid[s0:s0 + 256]))
        np.savez(g, ids=np.array(tid, dtype="U"), X=np.concatenate(X)); print("insert fit", b, flush=True)


# ------------------------------------------------------------------------------------------------------ arm d
class P6Setting:
    n_out = 2
    sets = ("fit", "val", "test")
    train_set, val_sets = "fit", ("val",)

    def __init__(self, b):
        self.KEY = f"p6_bias{bstr(b)}"
        self.part = {k: v.rename(columns={"label": "target"}).assign(image_id=v.image) for k, v in split(b).items()}
        self.used = used_flags(split(b)["fit"])

    def dataset(self, k, train):
        return P6Masked(self.part[k], train)

    def val_metric(self, preds):
        from sklearn.metrics import roc_auc_score
        y, p = preds["val"]
        return float(roc_auc_score(y, p[:, 1] - p[:, 0]))

    def labels(self, k):
        d = self.part[k]
        return d.target.to_numpy().astype(int), d[self.used].to_numpy().astype(int)


class P6Masked:
    """lesion-masked image (outside = mean colour): train = EPVT's training augmentation on the 256-px masked image; eval =
    the 224-px masked image (their Resize((224, 224)) of the original); ImageNet normalisation."""
    def __init__(self, rows, train):
        self.rows, self.train, self.c = rows.reset_index(drop=True), train, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        import torch
        import torchvision.transforms as T
        if self.c is None:
            self.c = cached(256 if self.train else 224)
            self.tf = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomResizedCrop(224, scale=(0.75, 1.0)),
                                 T.RandomRotation(45), T.ColorJitter(hue=0.1)]) if self.train else None
            self.norm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        I, M, pos = self.c; r = self.rows.iloc[k]; j = pos[r.image]
        x = Image.fromarray(masked(I[j], M[j]))
        if self.tf is not None:
            x = self.tf(x)
        return self.norm(x), torch.tensor(int(r.target))


def armd(biases, workers):
    import armd_train as A
    for b in biases:
        S = P6Setting(b); out = W / "armd" / S.KEY; out.mkdir(parents=True, exist_ok=True)
        if not (out / "TRAIN_DONE").exists():
            A.train(S, out, workers)
        if not (out / "features.npz").exists():
            A.features(S, out, workers)
        print("arm d", S.KEY, "done", flush=True)


# ------------------------------------------------------------------------------------------------------ heads, compare
def _select(Xs, ys, A, modes, grid):
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


def heads(biases):
    import candidates as CA
    import armd_train as A
    from sklearn.metrics import roc_auc_score
    from wtss.methods.insertion import fit_difference_subspace
    from wtss.utils import MEAN_RGB as MR
    assert tuple(MR) == MEAN_RGB
    zb = np.load(W / "dino518_masked_all.npz"); bpos = {i: j for j, i in enumerate(zb["ids"].astype(str))}
    rows, scores, sels = [], [], []
    for b in biases:
        sp = split(b); used = used_flags(sp["fit"]); Afl = sp["fit"][used].to_numpy().astype(int)
        ys = {s: sp[s].label.to_numpy() for s in sp}; B = bstr(b)
        arms = {}
        Xb = {s: zb["X"][[bpos[i] for i in sp[s].image]] for s in sp}
        arms["b_dino518_mask_cmc"] = [_select(Xb, ys, Afl, CA.CMC_MODES, CS)] * 5
        ins = np.load(W / f"dino518_insert_fit_bias{B}.npz"); assert list(ins["ids"]) == sp["fit"].image.tolist()
        er = fit_difference_subspace(Xb["fit"], ins["X"], energy=0.90, max_k=64, seed=0)
        arms["c_umte"] = [_select({s: er(v) for s, v in Xb.items()}, ys, Afl, (None,), CS)] * 5
        fa = [W / f"a_bias{B}_seed{s}.npz" for s in range(5)]
        if all(f.exists() for f in fa):
            arms["a_epvt_mask_cmc"] = [_select({k: np.load(f)[f"X_{k}"] for k in sp}, ys, Afl, CA.CMC_MODES, CS) for f in fa]
        fd = W / "armd" / f"p6_bias{B}" / "features.npz"
        if fd.exists():
            Fd = np.load(fd); arms["d_effb0_mask_cmc"] = [_select({k: Fd[k] for k in sp}, ys, Afl, CA.CMC_MODES, A.CS)] * 5
        for name, per in arms.items():
            for s, (sc, info, sel) in enumerate(per):
                if s == 0 or name == "a_epvt_mask_cmc":
                    rows.append({"bias": b, "arm": name, "epvt_seed": s if name == "a_epvt_mask_cmc" else "all", "flags": ",".join(used),
                                 "test_auroc": float(roc_auc_score(ys["test"], sc)), **info})
                    sels.append(pd.DataFrame(sel).assign(bias=b, arm=name, epvt_seed=s))
                scores.append(pd.DataFrame({"bias": b, "method": name, "seed": s, "image": sp["test"].image, "label": ys["test"], "score": sc}))
        for m in ("epvt", "erm"):
            for s in range(5):
                run = f"{D}/epvt/runs/" + (f"epvt_a36_bias{B}_1_seed{s}" if m == "epvt" else
                                           (f"erm_bias{B}_1" if (s == 0 and B == "1") else f"erm_bias{B}_1_seed{s}"))   # bias 1 ERM seed 0 = A33
                f = glob.glob(f"{run}/*__test_bias_{B}_1.csv.gz")
                if f:
                    e = pd.read_csv(f[0]).set_index("image").loc[sp["test"].image]
                    scores.append(pd.DataFrame({"bias": b, "method": m.upper(), "seed": s, "image": sp["test"].image, "label": ys["test"],
                                                "score": (e.logit1 - e.logit0).to_numpy()}))
    RES.mkdir(parents=True, exist_ok=True)
    tag = "_".join(bstr(b) for b in biases)
    pd.DataFrame(rows).to_csv(RES / f"arms_heads_bias{tag}.csv", index=False)
    pd.concat(scores).to_csv(RES / f"scores_test_bias{tag}.csv.gz", index=False)
    pd.concat(sels).to_csv(RES / f"selection_bias{tag}.csv", index=False)
    print(pd.DataFrame(rows)[["bias", "arm", "epvt_seed", "mode", "C", "val_auroc"]].to_string())


def compare(biases, final):
    from wtss.replication_stats import paired_bootstrap, holm, beaten
    tag = "_".join(bstr(b) for b in biases)
    s = pd.read_csv(RES / f"scores_test_bias{tag}.csv.gz")
    out = []
    for b in biases:
        x = s[s.bias == b]; ep = x[x.method == "EPVT"]
        for arm in [m for m in x.method.unique() if m not in ("EPVT", "ERM")]:
            per = {}
            for sd in range(5):
                a = x[(x.method == arm) & (x.seed == sd)]; e = ep[ep.seed == sd].set_index("image").loc[a.image]
                per[sd] = (a.image.tolist(), a.label.to_numpy(), a.score.to_numpy(), e.score.to_numpy())
            out.append({"bias": b, "arm": arm, **paired_bootstrap(per, "auroc")})
    o = pd.DataFrame(out).drop(columns=["seed_deltas"])
    if final:   # A39: Holm over the 8 comparisons (4 arms x 2 bias levels) only when both levels are in
        o["p_holm"] = holm(o.p_boot_two_sided.tolist()); o["beaten"] = [bool(beaten(r, r.p_holm)) for _, r in o.iterrows()]
    o.to_csv(RES / f"comparison_bias{tag}.csv", index=False)
    md = [f"# P6: our arms vs our EPVT (A39) — bias {', '.join(bstr(b) for b in biases)} — machine 2", "",
          "Test AUROC on `test_bias_<b>_1`, each arm scored once; crossed seed × image bootstrap vs EPVT's 5 seeds (A36 mixup).",
          ("Holm over the 8 A39 comparisons; 'beaten' = section 2." if final else
           "**Interim (bias 1 only): unadjusted; the A39 verdict uses both bias levels with Holm over 8.**"), "",
          "| bias | arm | arm mean | EPVT mean | Δ [95 % CI] | p (unadj.) |" + (" p Holm | beaten |" if final else ""),
          "|---|---|---|---|---|---|" + ("---|---|" if final else "")]
    for _, r in o.iterrows():
        md.append(f"| {bstr(r.bias)} | {r.arm} | {r.ours_mean:.3f} | {r.theirs_mean:.3f} | {r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] | "
                  f"{r.p_boot_two_sided:.4f} |" + (f" {r.p_holm:.4f} | {'yes' if r.beaten else 'no'} |" if final else ""))
    erm = s[s.method == "ERM"]
    if len(erm):
        from sklearn.metrics import roc_auc_score
        md += ["", "ERM (5 seeds, reference): " + ", ".join(f"bias {bstr(b)} {np.mean([roc_auc_score(g.label, g.score) for _, g in erm[erm.bias == b].groupby('seed')]):.3f}" for b in biases)]
    (RES / f"comparison_bias{tag}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "features_a", "features_bc", "armd", "heads", "compare"], required=True)
    ap.add_argument("--bias", nargs="+", type=float, default=[1.0]); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    {"cache": lambda: cache(a.bias), "features_a": lambda: features_a(a.bias), "features_bc": lambda: features_bc(a.bias),
     "armd": lambda: armd(a.bias, a.workers), "heads": lambda: heads(a.bias), "compare": lambda: compare(a.bias, a.final)}[a.stage]()
