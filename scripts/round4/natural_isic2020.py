"""R10 — external natural test: train on ISIC 2019, test on ISIC 2020 (docs/PREREGISTRATION_ROUND4.md).

  python scripts/round4/natural_isic2020.py --prepare      # dedup + counts only (no model)
  python scripts/round4/natural_isic2020.py --smoke        # quick check on a small subset
  python scripts/round4/natural_isic2020.py                # full run + analysis
Outputs: results/round4/natural_isic2020/{dedup.csv, counts.json, natural_metrics.csv, natural_boot.csv,
operating_points.csv, SUMMARY.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

from wtss import paths  # noqa: E402

EXT = K.REF_ROOT / "external_isic2020"
W20 = paths.DATA / "external" / "isic2020_work" / "cache_518"
PFX = "i20_"
N_BOOT = 10000
N_BOOT_OP = 2000
PAIRS = [("mask", "erm"), ("balanced", "mask"), ("mte_balanced", "mask"), ("mte", "mask")]


class TwoCache:
    """ISIC 2019 cache for plain IDs, ISIC 2020 cache for IDs with the i20_ prefix."""

    def __init__(self, c19, c20):
        self.c19, self.c20 = c19, c20

    def get(self, i):
        i = str(i)
        return self.c20.get(i[len(PFX):]) if i.startswith(PFX) else self.c19.get(i)


def phash_all(cache, ids, threads=16):
    import imagehash

    def one(i):
        return int(str(imagehash.phash(Image.fromarray(np.asarray(cache.get(i)[0])), hash_size=8, highfreq_factor=4)), 16)
    with ThreadPoolExecutor(threads) as ex:
        return np.array(list(ex.map(one, ids)), dtype=np.uint64)


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def load_all(smoke=False):
    """(ISIC 2019 table, ISIC 2020 table, combined cache, ISIC 2019 feature dir)."""
    from wtss.experiments.real_traps import RealCache
    rn = K.load_script("rn", "scripts/run_natural.py")
    c19, cache19, fdir19, _ = rn.load("isic_HAM")  # every ISIC 2019 image; the source split is not used here
    c19 = c19.copy()
    c19["image_id"] = c19.image_id.astype(str)
    tau = json.loads((EXT / "segmenter.json").read_text())["tau_pred_hair_px"]
    d = pd.read_csv(EXT / "cohort.csv")
    d = d[d.lesion_frac > 0].copy()
    d["name"] = d.image_name.astype(str)
    d["image_id"] = PFX + d.name
    d["y"] = d.target.astype(int)
    d["a"] = ((d.hair_px > tau) & (d.r >= 0.5)).astype(int)
    d["source"], d["group"] = "ISIC2020", d.patient_id.astype(str)
    cache20 = RealCache(W20, roi_file="roi.npy", art_file="hair.npy")
    d = d[d.name.isin(set(cache20.index))].reset_index(drop=True)
    if smoke:
        d = pd.concat([d[d.y == 1].head(60), d[d.y == 0].head(240)]).reset_index(drop=True)
        g = c19.group.drop_duplicates().sample(frac=1, random_state=0)
        c19 = c19[c19.group.isin(set(g.head(max(1, len(g) // 20))))].reset_index(drop=True)
    return rn, c19, d, TwoCache(cache19, cache20), fdir19, tau


def prepare(out: Path, smoke=False):
    rn, c19, d, cache, fdir19, tau = load_all(smoke)
    f = out / "dedup.csv"
    if not f.exists():
        K.log(step="phash", n19=len(c19), n20=len(d))
        h19 = phash_all(cache, c19.image_id.tolist())
        h20 = phash_all(cache, d.image_id.tolist())
        best_d, best_j = np.full(len(h20), 64, np.int32), np.zeros(len(h20), np.int64)
        for k in range(0, len(h20), 256):
            x = popcount64(h20[k:k + 256, None] ^ h19[None, :])
            j = x.argmin(1)
            best_d[k:k + 256], best_j[k:k + 256] = x[np.arange(len(j)), j], j
        same_name = d.name.isin(set(c19.image_id))
        drop = (best_d <= 8) | same_name.to_numpy()
        pd.DataFrame({"image_id": d.image_id, "nearest_isic2019": c19.image_id.to_numpy()[best_j], "hamming": best_d,
                      "same_name": same_name, "dropped": drop}).to_csv(f, index=False)
    dd = pd.read_csv(f)
    keep = set(dd[~dd.dropped].image_id)
    test = d[d.image_id.isin(keep)].reset_index(drop=True)
    rate = lambda t: {f"P(a|y={y})": float(t[t.y == y].a.mean()) for y in (1, 0)}
    cnt = {"tau_pred_hair_px": tau, "isic2020_with_lesion_mask": int(len(d)), "dropped_near_duplicates": int(dd.dropped.sum()),
           "dropped_same_name": int(dd.same_name.sum()), "test_images": int(len(test)), "test_melanomas": int(test.y.sum()),
           "test_patients": int(test.group.nunique()), "isic2019_train_val_images": int(len(c19)),
           "isic2019_melanomas": int(c19.y.sum()), "isic2019_in_lesion_hair": rate(c19), "isic2020_in_lesion_hair": rate(test)}
    (out / "counts.json").write_text(json.dumps(cnt, indent=1))
    print(json.dumps(cnt, indent=1), flush=True)
    return rn, c19, test, cache, fdir19


def cross_group(y, a, p):
    from wtss.stats import safe_auc
    out = []
    for hp, hn in ((0, 1), (1, 0)):
        s = ((y == 1) & (a == hp)) | ((y == 0) & (a == hn))
        out.append(safe_auc(y[s], p[s]))
    return float(np.nanmin(out))


def _boot(job):
    q, a1, a0, seed = job
    from wtss import stats_crossed as X
    return X.hierarchical_paired_bootstrap(q, a1, a0, "clean", N_BOOT, seed)


def op_crossed(p, arm, ref, seed, hard_pos_a=0):
    """Crossed bootstrap (seeds x Poisson image weights) of arm - ref at OP1-OP5; thresholds re-estimated on the
    weighted ISIC 2019 validation images of each seed (scripts/analysis/operating_points.py, crossed)."""
    op = K.load_script("op", "scripts/analysis/operating_points.py")
    ids = sorted(p.image_id.astype(str).unique()); pos = {k: j for j, k in enumerate(ids)}
    S = {}
    for s, q in p.groupby("seed"):
        g = lambda m, e: q[(q.method == m) & (q.env == e)].sort_values("image_id")
        va, vr, ta, tr = g(arm, "val_groups"), g(ref, "val_groups"), g(arm, "clean"), g(ref, "clean")
        S[s] = dict(vy=va.y.to_numpy(), vpa=va.prob.to_numpy(), vpr=vr.prob.to_numpy(), vi=np.array([pos[i] for i in va.image_id.astype(str)]),
                    ty=ta.y.to_numpy(), tpa=ta.prob.to_numpy(), tpr=tr.prob.to_numpy(), ti=np.array([pos[i] for i in ta.image_id.astype(str)]),
                    ta=ta.artifact_present.to_numpy())
    seeds = sorted(S); rng = np.random.default_rng(seed)
    keys = ("sens", "spec", "sens_conflict")
    res = {o: {k: [] for k in keys} for o in op.OPS}
    for _ in range(N_BOOT_OP):
        Wt = rng.poisson(1.0, len(ids)).astype(float)
        pick = rng.choice(seeds, len(seeds), replace=True)
        acc = {o: {k: [] for k in keys} for o in op.OPS}
        for s in pick:
            d = S[s]; wv, wt = Wt[d["vi"]], Wt[d["ti"]]
            y, t = d["ty"], d["ta"]
            hp = (y == 1) & (t == hard_pos_a)
            for o, (kind, tg) in op.OPS.items():
                vals = []
                for vp, tp in ((d["vpa"], d["tpa"]), (d["vpr"], d["tpr"])):
                    th = op.threshold_w(d["vy"], vp, wv, kind, tg)
                    pr = tp >= th
                    vals.append({"sens": np.sum(wt * pr * (y == 1)) / max(np.sum(wt * (y == 1)), 1e-9),
                                 "spec": np.sum(wt * ~pr * (y == 0)) / max(np.sum(wt * (y == 0)), 1e-9),
                                 "sens_conflict": np.sum(wt * pr * hp) / max(np.sum(wt * hp), 1e-9)})
                for k in keys:
                    acc[o][k].append(vals[0][k] - vals[1][k])
        for o in op.OPS:
            for k in keys:
                res[o][k].append(np.mean(acc[o][k]))
    per = op.per_seed(p[p.method.isin([arm, ref])], hard_pos_a).groupby(["op", "method"])[list(keys)].mean()
    rows = []
    for o in op.OPS:
        for k in keys:
            lo, hi = np.percentile(res[o][k], [2.5, 97.5])
            va, vr = per.loc[(o, arm), k], per.loc[(o, ref), k]
            rows.append({"arm": arm, "ref": ref, "op": o, "metric": k, "arm_value": va, "ref_value": vr, "delta": va - vr,
                         "ci95_lo": lo, "ci95_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": N_BOOT_OP, "boot_seed": seed})
    return rows


def analyse(out: Path, c19, test):
    from wtss.stats import safe_auc
    p = pd.read_csv(out / "predictions.csv.gz")
    p["image_id"] = p.image_id.astype(str)
    te = p[p.env == "clean"].copy()
    rows = []
    for (m, s), q in te.groupby(["method", "seed"]):
        y, pr, ai = q.y.to_numpy(), q.prob.to_numpy(), q.artifact_present.to_numpy()
        hard = ((y == 1) & (ai == 0)) | ((y == 0) & (ai == 1))
        rows.append({"method": m, "seed": s, "auc": safe_auc(y, pr), "cross_group": cross_group(y, ai, pr),
                     "auc_in_lesion_hair": safe_auc(y[ai == 1], pr[ai == 1]), "auc_hard": safe_auc(y[hard], pr[hard]),
                     "auc_easy": safe_auc(y[~hard], pr[~hard])})
    m = pd.DataFrame(rows)
    m.to_csv(out / "natural_metrics_per_seed.csv", index=False)
    mm = m.groupby("method").mean(numeric_only=True).drop(columns="seed")
    mm.to_csv(out / "natural_metrics.csv")
    print(mm.round(3).to_string(), flush=True)
    hard = ((te.y == 1) & (te.artifact_present == 0)) | ((te.y == 0) & (te.artifact_present == 1))
    jobs, keys = [], []
    for name, sel in (("hard", hard), ("easy", ~hard), ("all", hard | ~hard)):
        for k, (a1, a0) in enumerate(PAIRS):
            if {a1, a0} <= set(te.method):
                q = te[sel & te.method.isin([a1, a0])][["seed", "method", "env", "image_id", "y", "prob"]]
                jobs.append((q, a1, a0, 20261401 + 10 * k + ("hard", "easy", "all").index(name)))
                keys.append((name, a1, a0))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_boot, jobs))
    b = pd.DataFrame([{"subset": n, "arm": a1, "ref": a0, "estimate": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"],
                       "ci95_hi": r["ci95_hi"], "p_boot_two_sided": r["p_boot_two_sided"], "boot_seed": j[3],
                       "estimator": r.get("estimator", "")} for (n, a1, a0), r, j in zip(keys, res, jobs)])
    b.to_csv(out / "natural_boot.csv", index=False)
    print(b.round(3).to_string(), flush=True)
    pv = p[p.env.isin(["clean", "val_groups"])]
    ops = []
    for k, (arm, ref) in enumerate([("mask", "erm"), ("balanced", "mask")]):
        if {arm, ref} <= set(pv.method):
            ops += op_crossed(pv, arm, ref, 20261501 + k)
    o = pd.DataFrame(ops)
    o.to_csv(out / "operating_points.csv", index=False)
    return mm, b, o


def write_summary(out: Path, mm, b, o):
    cnt = json.loads((out / "counts.json").read_text())
    g = lambda s, a1, a0: b[(b.subset == s) & (b.arm == a1) & (b.ref == a0)].iloc[0]
    h10a, h10c = g("hard", "mask", "erm"), g("hard", "balanced", "mask")
    op5 = o[(o.arm == "mask") & (o.ref == "erm") & (o.op == "OP5_sens0.90") & (o.metric == "sens_conflict")].iloc[0]
    v = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"
    L = ["# R10 — external natural test: ISIC 2019 -> ISIC 2020 (DINOv2 ViT-B/14 @518)", "",
         f"Environment: {json.dumps(K.env_info())}", "", "Counts:", "", "```", json.dumps(cnt, indent=1), "```", "",
         "Hard pairs: melanomas without in-lesion hair vs benign lesions with in-lesion hair (conflict with the training association).", "",
         K.md_table(mm.reset_index().round(3)), "",
         "Crossed bootstrap (5 seeds; one Poisson weight per ISIC 2020 image):", "",
         K.md_table(b.assign(value=[f"{r['estimate']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]" for r in b.to_dict("records")])
                    [["subset", "arm", "ref", "value"]]), "",
         f"- **H10a** hard-pair AUROC(mask) - AUROC(ERM) < 0: {h10a.estimate:+.3f} [{h10a.ci95_lo:+.3f}, {h10a.ci95_hi:+.3f}] -> "
         f"**{v(h10a.ci95_hi < 0)}**",
         f"- **H10b** OP5 sensitivity among melanomas without in-lesion hair, mask - ERM < 0: {op5.delta:+.3f} "
         f"[{op5.ci95_lo:+.3f}, {op5.ci95_hi:+.3f}] -> **{v(op5.ci95_hi < 0)}**",
         f"- **H10c** hard-pair AUROC(balanced) - AUROC(mask) > 0: {h10c.estimate:+.3f} [{h10c.ci95_lo:+.3f}, {h10c.ci95_hi:+.3f}] -> "
         f"**{v(h10c.ci95_lo > 0)}**", "",
         "Operating points (thresholds from ISIC 2019 validation only):", "",
         K.md_table(o.assign(value=[f"{r['arm_value']:.3f} vs {r['ref_value']:.3f}: {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"
                                    for r in o.to_dict("records")])[["arm", "ref", "op", "metric", "value"]]), ""]
    (out / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_jobs", type=int, default=4)
    a = ap.parse_args()
    out = K.OUT / ("_smoke" if a.smoke else "") / "natural_isic2020"
    out.mkdir(parents=True, exist_ok=True)
    rn, c19, test, cache, fdir19 = prepare(out, a.smoke)
    if a.prepare:
        return
    import torch
    from wtss.experiments.real_traps import make_renderers
    from wtss.experiments.spec_traps import run_spec
    from wtss.synthetic import draw_generic_artifact
    from wtss.utils import stable_int
    envs = {}
    tcols = test[rn.COLS].reset_index(drop=True)
    for s in rn.SEEDS[:1] if a.smoke else rn.SEEDS:
        g = c19.group.unique()
        val_g = {x for x in g if stable_int("natural_val", s, x) % 5 == 0}
        va, tr = c19[c19.group.isin(val_g)], c19[~c19.group.isin(val_g)]
        e = {"train_corr": tr, "val_clean": va, "val_groups": va, "train_all": tr}
        for k, dd in e.items():
            envs[("natural", s, 0, k)] = dd[rn.COLS].reset_index(drop=True)
        for k in ("test_corr", "test_rev", "clean"):
            envs[("natural", s, 0, k)] = tcols
    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    dev = torch.device(a.device)
    fdir = K.FEAT / ("_smoke" if a.smoke else "") / "natural_isic2020" / K.BDIR
    pool = K.pool_of(envs, ("natural",))
    rend = make_renderers(cache, 518, [], ins)
    fname = lambda v: f"{v}_generic.npz" if v in ("insert", "mask_insert") else f"{v}.npz"
    strip = lambda i: i[len(PFX):] if i.startswith(PFX) else None
    f20 = paths.CACHE / "features" / "isic2020" / K.BDIR
    srcs = {v: [(fdir19 / fname(v), lambda i: None if i.startswith(PFX) else i)] +
            ([(f20 / fname(v), strip)] if v in ("erm", "mask") else []) for v in ("erm", "mask", "mask_insert")}
    if not (out / "predictions.csv.gz").exists():
        K.assemble_views(pool, ("erm", "mask", "mask_insert"), rend, fdir, srcs, dev, fname=fname, workers=a.workers)
        run_spec(envs, cache, K.BACKBONE, out, fdir, [], arms=rn.ARMS, traps=("natural",), device=dev, insert_fn=ins,
                 insert_tag="_generic", folds=[0], save_val=True, workers=a.workers, n_jobs=a.n_jobs)
    mm, b, o = analyse(out, c19, test)
    write_summary(out, mm, b, o)


if __name__ == "__main__":
    main()
