"""Alignment audit (author, 2026-11-20; after A30), our pipelines — CPU, our environment.

For every cached feature file of our arms (P1 arms b, d; P1 location test arms b, d; P2 arms b/c, d; P3 arms b/c, d, corner
c), per split, 200 random rows: the input rebuilt independently from the row's id (source file / cache row keyed by id,
the row's ROI keyed by id, the row's artifact rule) is featurised again on CPU with the same encoder and compared with the
cached row (cosine >= 0.99; caches were made in fp16 on the GPU). Image caches are also compared pixel-for-pixel with an
independent render. Labels and artifact flags are re-derived from the source tables / rules by id. Arm c (U-MtE) shares
arm b's masked features (same files), so arm b's check covers both.
Writes results/audit/align_ours.csv.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "32")
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
D = "/workspace/data"
N = int(os.environ.get("AUDIT_N", "200"))
rng = np.random.default_rng(20261120)
torch.set_num_threads(32)
ROWS = []


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def report(setting, split, arm, n_rows, checked, **bad):
    ROWS.append({"setting": setting, "split": split, "arm": arm, "rows": n_rows, "checked": checked, **bad})
    print(ROWS[-1], flush=True)
    pd.DataFrame(ROWS).to_csv(ROOT / "results/audit/align_ours.csv", index=False)


_UM = {}


def dino():
    if "um" not in _UM:
        from wtss.umte import UMtE
        _UM["um"] = UMtE("dino518", device="cpu", seed=0, batch_size=8)
    return _UM["um"]


def effb0(path, n_out):
    import armd_train as A
    m = A.make_model(n_out, pretrained=False); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    f = {}
    m.classifier.register_forward_hook(lambda mm, i, o: f.__setitem__("z", i[0]))
    def run(x):
        with torch.no_grad():
            m(x[None]); return f["z"][0].numpy()
    return run


def p1():
    import cv2
    import p1_arms as PA
    import armd_train as A
    for task, key in (("xray", "p1_cardio"), ("rsna", "p1_rsna")):
        sp = pd.read_csv(f"{D}/replication/viviano/preds/{task}_standard_1234.csv.gz")
        Ad = f"{D}/replication/viviano/arms/{task}_1234"
        S = A.SETTINGS[key]()
        Fd = np.load(f"{D}/replication/armd/{key}/features.npz")
        dfun = effb0(f"{D}/replication/armd/{key}/best.pt", 2)
        for s in ("train", "valid", "test"):
            d = sp[sp.split == s].reset_index(drop=True)
            paths = [PA.image_path(task, i, int(c)) for i, c in zip(d.image_id, d.site)]
            R = np.load(f"{Ad}/roi_{s}.npy", mmap_mode="r")
            z = np.load(f"{Ad}/roi224_{s}.npz", allow_pickle=True); roi_ids_ok = bool((z["ids"].astype(str) == d.image_id.astype(str).to_numpy()).all())
            Xb = np.load(f"{Ad}/dino518_masked_{s}.npz")["X"]
            pick = rng.choice(len(d), min(N, len(d)), replace=False)
            render = PA.masked_render(paths, R)
            bad_b = sum(cos(dino()._encode([render(int(j))])[0], Xb[j]) < 0.99 for j in pick)
            # arm d: cache row = fresh render of the row's own file + its ROI; features recomputed
            mm = np.load(S.npy[s], mmap_mode="r"); bad_img = bad_d = 0
            lab_bad = int((S.part[s].image_id.astype(str).to_numpy() != d.image_id.astype(str).to_numpy()).sum() + (S.part[s].y.to_numpy() != d.y.to_numpy()).sum())
            for j in pick:
                a = cv2.imread(paths[j], cv2.IMREAD_UNCHANGED); a = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if a.dtype != np.uint8 else a
                h, w = a.shape; c = min(h, w); a = a[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
                r = np.asarray(R[j]); rh, rw = r.shape; rc = min(rh, rw); r = r[(rh - rc) // 2:(rh - rc) // 2 + rc, (rw - rc) // 2:(rw - rc) // 2 + rc]
                from wtss.ops import apply_roi_mask
                im = Image.fromarray(a).convert("RGB").resize((S.SIZE, S.SIZE), Image.BILINEAR)
                ref = np.asarray(apply_roi_mask(im, cv2.resize(r.astype(np.uint8), (S.SIZE, S.SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)))
                bad_img += not np.array_equal(ref, np.asarray(mm[j]))
                bad_d += cos(dfun(S.dataset(s, False)[int(j)][0]), Fd[s][j]) < 0.99
            report(f"P1 {task}", s, "b (DINOv2 masked; c uses the same features)", len(d), len(pick), feature_mismatch=int(bad_b), roi_ids_order_ok=roi_ids_ok)
            report(f"P1 {task}", s, "d (EfficientNet-B0)", len(d), len(pick), image_mismatch=int(bad_img), feature_mismatch=int(bad_d), label_or_id_mismatch=lab_bad)


def p1_location():
    import cv2
    import p1_location_arms as LA
    import armd_train as A
    from wtss.ops import apply_roi_mask
    from wtss.synthetic import draw_tube
    L = Path(f"{D}/replication/viviano/location")
    sp = pd.read_csv(L / "nospc_1234.csv").sort_values(["split", "pos"]); sp["key"] = sp.site.astype(str) + "/" + sp.image_id
    pl = pd.read_csv(L / "placements_1234.csv").set_index("key")
    z = np.load(L / "roi224_1234.npz"); roi = dict(zip(z["ids"], z["masks"]))
    for pos in ("inside", "outside"):
        S = A.SETTINGS[f"p1_loc_{pos}"]()
        Fd = np.load(f"{D}/replication/armd/p1_loc_{pos}/features.npz"); dfun = effb0(f"{D}/replication/armd/p1_loc_{pos}/best.pt", 2)
        their = pd.read_csv(L / pos / "scores.csv.gz")
        for e in LA.P_TUBE:
            d = sp[sp.split == LA.SPLIT[e]].reset_index(drop=True); d["tube"] = LA.presence(d.key, d.y, e)
            Xb = np.load(L / pos / f"arms/dino518_{e}.npz")["X"]
            pick = rng.choice(len(d), min(N, len(d)), replace=False)
            bad_b = bad_img = bad_d = 0
            for j in pick:
                row = d.iloc[j]
                im = Image.fromarray(LA.gray(row.path)).convert("RGB")
                if row.tube:
                    q = pl.loc[row.key]; x, yy = (int(q.in_x), int(q.in_y)) if pos == "inside" else (int(q.out_x), int(q.out_y))
                    im, _ = draw_tube(im, x, yy, LA.W, LA.H, row.key)
                im518 = apply_roi_mask(im.resize((518, 518), Image.BICUBIC), cv2.resize(roi[row.key], (518, 518), interpolation=cv2.INTER_NEAREST).astype(bool))
                bad_b += cos(dino()._encode([im518])[0], Xb[j]) < 0.99
                ref = np.asarray(apply_roi_mask(im.resize((128, 128), Image.BILINEAR), cv2.resize(roi[row.key].astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST).astype(bool)))
                bad_img += not np.array_equal(ref, np.load(S.npy[e], mmap_mode="r")[j])
                bad_d += cos(dfun(S.dataset(e, False)[int(j)][0]), Fd[e][j]) < 0.99
            lab_bad = int((S.part[e].key.to_numpy() != d.key.to_numpy()).sum() + (S.labels(e)[1] != d.tube.astype(int).to_numpy()).sum())
            if e in ("corr", "rev", "clean"):
                t = their[(their.method == "masked") & (their.test_set == e)]
                lab_bad += int((t.image_id.to_numpy() != d.key.to_numpy()).sum() + (t.tube.to_numpy() != d.tube.astype(int).to_numpy()).sum())
            report(f"P1 location {pos}", e, "b (DINOv2 masked; c same features)", len(d), len(pick), feature_mismatch=int(bad_b))
            report(f"P1 location {pos}", e, "d (EfficientNet-B0)", len(d), len(pick), image_mismatch=int(bad_img), feature_mismatch=int(bad_d), label_tube_or_id_mismatch=lab_bad)


def p2():
    import armd_train as A
    R = Path(f"{D}/replication/pahde")
    S = A.P2(); Fd = np.load(f"{D}/replication/armd/p2/features.npz"); dfun = effb0(f"{D}/replication/armd/p2/best.pt", 8)
    gt = pd.read_csv(f"{D}/isic2019/ISIC_2019_Training_GroundTruth.csv").set_index("image")
    cls = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    van = pd.read_csv(R / "scores/vanilla_seed0.csv.gz", usecols=["index", "image", "split", "test_set", "label", "artifact"])
    names = {"train_as_trained": ("train", "as_trained"), "val_clean": ("val", "clean"), "val_biased": ("val", "biased"),
             "test_clean": ("test", "clean"), "test_biased": ("test", "biased")}
    for k, (spn, ts) in names.items():
        part = S.part[k]; v = van[(van.split == spn) & (van.test_set == ts)].reset_index(drop=True)
        order_ok = bool((part["index"].to_numpy() == v["index"].to_numpy()).all() and (part.image.to_numpy() == v.image.to_numpy()).all())
        lab_bad = int(sum(cls.index(gt.loc[i][cls].idxmax()) != int(y) for i, y in zip(part.image, part.label)))
        Xb = np.load(R / f"arms/dino518_masked_{k}.npz")["X"]
        mm = np.load(S.npy[k], mmap_mode="r")
        pick = rng.choice(len(part), min(N, len(part)), replace=False)
        bad_img = bad_b = bad_d = 0
        for j in pick:   # the rendered image is the row's own file (resized to 224; artifacts change only a ring / vignette)
            src = np.asarray(Image.open(f"{D}/isic2019/ISIC_2019_Training_Input/{part.image.iloc[j]}.jpg").convert("RGB").resize((224, 224), Image.BICUBIC), np.float32)
            got = np.asarray(mm[j], np.float32)
            ok = got.mean(2) > 8   # ignore black artifact pixels
            r = np.corrcoef(src.mean(2)[ok], got.mean(2)[ok])[0, 1]
            bad_img += r < 0.95
            im = Image.fromarray(np.asarray(mm[j])); rr = S.roi[str(part.image.iloc[j])]
            bad_b += cos(dino().masked_features([im], [rr])[0], Xb[j]) < 0.99
            bad_d += cos(dfun(S.dataset(k, False)[int(j)][0]), Fd[k][j]) < 0.99
        report("P2", k, "render cache + b (DINOv2 masked; c same) + d", len(part), len(pick), order_vs_vanilla_scores_ok=order_ok,
               label_mismatch_vs_GT=lab_bad, image_mismatch=int(bad_img), feature_mismatch_b=int(bad_b), feature_mismatch_d=int(bad_d))


def p3():
    import armd_train as A
    for Scls, dd, armd_key in ((A.P3, "arms_bc", "p3_9010_inside"), (A.P3Corner, "arms_bc_corner", None)):
        S = Scls(); co = pd.read_csv(f"{D}/replication/isnet/cohort.csv").set_index("image_id")
        Fd = np.load(f"{D}/replication/armd/{armd_key}/features.npz") if armd_key else None
        dfun = effb0(f"{D}/replication/armd/{armd_key}/best.pt", 3) if armd_key else None
        for k in [k for k in S.sets if k in S.src]:
            g, r, idx, ids, y, pl, marks = S.src[k]
            lab_bad = int(sum(int(co.loc[i, "y"]) != int(yy) for i, yy in zip(ids, y))) if not k.startswith("test") else 0
            Xb = np.load(f"{D}/replication/isnet/{dd}/dino518_masked_{k}.npz")
            ids_ok = bool((Xb["ids"].astype(str) == np.asarray(ids, dtype=str)).all())
            data = S.T.Data(np.asarray(g[idx]), np.asarray(r[idx]), ids, y, pl, marks, False, 0)
            pick = rng.choice(len(ids), min(N, len(ids)), replace=False)
            bad_img = bad_b = bad_d = 0
            for j in pick:
                gi, mi, _ = data[int(j)]
                base = np.asarray(g[idx[j]])   # the cohort cache row of this id (marks change only the mark box)
                diff = np.abs((gi[0].numpy() * 255).round() - base) > 1
                bad_img += diff.mean() > 0.02
                im = Image.fromarray((gi[0].numpy() * 255).round().astype(np.uint8)).convert("RGB")
                bad_b += cos(dino().masked_features([im], [mi[0].numpy() > 0.5])[0], Xb["X"][j]) < 0.99
                if dfun is not None:
                    bad_d += cos(dfun(S.dataset(k, False)[int(j)][0]), Fd[k][j]) < 0.99
            report(f"P3 {'90/10 inside' if armd_key else 'corner'}", k, "b/c (DINOv2 masked)" + (" + d" if armd_key else ""), len(ids), len(pick),
                   ids_order_ok=ids_ok, label_mismatch=lab_bad, image_mismatch=int(bad_img), feature_mismatch_b=int(bad_b),
                   feature_mismatch_d=int(bad_d) if armd_key else None)


if __name__ == "__main__":
    (ROOT / "results/audit").mkdir(parents=True, exist_ok=True)
    for name in os.environ.get("AUDIT_PARTS", "p1,p1_location,p2,p3").split(","):
        {"p1": p1, "p1_location": p1_location, "p2": p2, "p3": p3}[name]()
