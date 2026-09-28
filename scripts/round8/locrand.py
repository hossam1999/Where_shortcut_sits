"""Round 8, N3 locrand: pasted real artifact instances (docs/PREREGISTRATION_ROUND8.md, sections 3 and 6).

  python scripts/round8/locrand.py --cohort thyroid --step placements   # CPU: K donor versions per recipient
  python scripts/round8/locrand.py --cohort thyroid --step extract      # GPU: masked features of every pasted version
  python scripts/round8/locrand.py --cohort thyroid --step probe        # CPU: registered pasted-vs-real probe
  (--smoke: 24 recipients, K = 2, output under $WTSS_CACHE/features/round8/_smoke and results/round8/_smoke)

Recipients are the cohort's artifact-free images (thyroid / ovary: no detected marker pixel; capsule: contamination
< 3 %; ISIC: ≤ 30 native hair pixels). For each recipient, K = 4 versions: version k takes the k-th distinct donor in a
stable random order (donors = artifact-bearing images whose largest grouped artifact component is within the size
limits of wtss.transplant) with a dihedral transform that admits both an in-ROI position (overlap ≥ 0.95) and an
out-of-ROI position (overlap 0) inside the field of view (the placement rule of Stage 4). Each version is rendered
twice, pasted inside and pasted outside, and masked. A training set may use only versions whose donor is one of its own
training images, so no test or validation image contributes pixels to training.
Nothing here trains a classifier on labels or scores a test set (the probe uses artifact status only).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r8(name):
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


C = _r8("common")
K_VERSIONS = 4
BDIR = {"dino518": "dinov2_b14_518"}
COHORTS = ("thyroid", "capsule", "ovary", "isic2019")


def cohort_info(name: str):
    """(table with image_id, y, group, recipient, donor, present, r), RealCache, artifact kind)."""
    from wtss import paths
    from wtss.experiments.real_traps import RealCache
    rt = C.load_round4_common().load_script("rt", "scripts/run_thyroid_traps.py")
    if name in ("thyroid", "ovary"):
        c = rt.cohort() if name == "thyroid" else rt.ovary_cohort()
        cache = RealCache((rt.T if name == "thyroid" else rt.OV) / "cache_518", roi_file="roi.npy", art_file="marker.npy")
        c["recipient"], c["donor"], kind = c.marker_px == 0, c.marker_px >= 15, "caliper"
    elif name == "capsule":
        c = rt.capsule_cohort()
        cache = RealCache(rt.CAP / "cache_518", roi_file="roi.npy", art_file="contam.npy")
        c["recipient"], c["donor"], kind = c.contam_frac < 0.03, c.contam_frac >= 0.10, "debris"
    elif name == "isic2019":
        c = pd.read_csv(paths.DATA / "isic2019" / "prepared" / "cohort_spec.csv")
        c = c[c.qc_ok.astype(bool)].copy()
        c["r"] = c.r_spec
        cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
        c["recipient"], c["donor"], kind = c.hair_px_native <= 30, c.hair_px_native > 30, "hair"
    else:
        raise ValueError(name)
    c["image_id"] = c.image_id.astype(str)
    c["group"] = c.group.astype(str)
    c["present"] = c.donor
    return c.reset_index(drop=True), cache, kind


def feat_dir(name: str, smoke: bool, encoder: str = "dino518") -> Path:
    return C.FEAT / ("_smoke" if smoke else "") / "locrand" / name / BDIR[encoder]


# ------------------------------------------------------------------------------------------------ placements
_G: dict = {}


def _instance(d):
    from wtss.transplant import extract_instance
    rgb, _, art = _G["cache"].get(d)
    return d, extract_instance(art, rgb, _G["kind"])


def _place(i):
    from wtss.transplant import _window_fraction, dihedral, field_of_view, FOV_MIN, IN_MIN
    from wtss.utils import stable_int
    rgb, roi, _ = _G["cache"].get(i)
    fov = field_of_view(rgb)
    roi = (roi > 0).astype(np.uint8)
    donors, inst, k_max = _G["donors"], _G["inst"], _G["k"]
    rng = np.random.default_rng(stable_int("r8_locrand_order", i))
    order = [donors[j] for j in rng.permutation(len(donors))[:200]]
    out, used = [], set()
    for d in order:
        if len(out) >= k_max:
            break
        if d == i or d in used:
            continue
        for op in rng.permutation(8):
            m = dihedral(inst[d]["mask"], int(op))
            if m.shape[0] >= roi.shape[0] or m.shape[1] >= roi.shape[1]:
                continue
            f_roi, f_fov = _window_fraction(roi, m), _window_fraction(fov, m)
            ok = f_fov >= FOV_MIN
            pin, pout = np.argwhere(ok & (f_roi >= IN_MIN)), np.argwhere(ok & (f_roi <= 1e-6))
            if len(pin) and len(pout):
                prng = np.random.default_rng(stable_int("r8_locrand_pos", i, d, int(op)))
                yi, xi = pin[prng.integers(len(pin))]
                yo, xo = pout[prng.integers(len(pout))]
                out.append({"k": len(out), "donor": d, "op": int(op), "in": [int(xi), int(yi)], "out": [int(xo), int(yo)]})
                used.add(d)
                break
    return i, out


def placements(name: str, smoke: bool) -> dict:
    fd = feat_dir(name, smoke)
    pf = fd.parent / "placements.json"
    if pf.exists():
        return json.loads(pf.read_text())
    c, cache, kind = cohort_info(name)
    rec = c[c.recipient].image_id.tolist()
    don = c[c.donor].image_id.tolist()
    if smoke:  # the recipients the smoke training sets contain (round-4 smoke subset of the trap cohort)
        trap_name = {"isic2019": "isic"}.get(name, name)
        R4 = C.load_round4_common()
        sub = R4.smoke_subset(R4.trap_cohort(trap_name)["c"], ("trapA", "trapB"), n=40, seed=0)
        keep = set(sub.image_id.astype(str))
        rec, don = [i for i in rec if i in keep][:60], don[:300]
    _G.update(cache=cache, kind=kind)
    with mp.get_context("fork").Pool(min(32, C.load_round4_common().n_cpus() * 2)) as pool:
        inst = dict(pool.map(_instance, don, chunksize=16))
    inst = {d: v for d, v in inst.items() if v is not None}
    _G.update(inst=inst, donors=sorted(inst), k=2 if smoke else K_VERSIONS)
    C.log(cohort=name, recipients=len(rec), valid_donors=len(inst))
    with mp.get_context("fork").Pool(32) as pool:
        res = dict(pool.map(_place, rec, chunksize=4))
    pl = {i: v for i, v in res.items() if v}
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(pl))
    C.log(cohort=name, placed=len(pl), with_all_versions=sum(len(v) == _G["k"] for v in pl.values()))
    return pl


def placement_stats(name: str, smoke: bool) -> dict:
    pl = placements(name, smoke)
    n = [len(v) for v in pl.values()]
    return {"cohort": name, "recipients_placed": len(pl), "versions_mean": float(np.mean(n)) if n else 0.0,
            "donors_used": len({p["donor"] for v in pl.values() for p in v})}


# ------------------------------------------------------------------------------------------------ rendering
def make_paste_renderer(cache, kind: str, pl: dict, k: int, loc: str, out_size: int, masked: bool = True):
    """image_id -> PIL image of recipient i with its version-k instance pasted at the in/out position (then masked)."""
    from PIL import Image
    from wtss.ops import apply_roi_mask
    from wtss.transplant import composite, extract_instance
    memo: dict = {}

    def inst(d):
        if d not in memo:
            rgb, _, art = cache.get(d)
            memo[d] = extract_instance(art, rgb, kind)
        return memo[d]

    def render(i):
        v = pl[str(i)][k]
        rgb, roi, _ = cache.get(str(i))
        img, _ = composite(Image.fromarray(rgb), inst(v["donor"]), v["op"], *v[loc])
        if masked:
            img = apply_roi_mask(img, roi)
        return img if img.size[0] == out_size else img.resize((out_size, out_size), Image.BICUBIC)

    return render


def extract(name: str, smoke: bool, encoder: str = "dino518", batch_size: int = 64, workers: int = 6):
    import torch
    from wtss.backbones import BACKEND_SIZE, load_backend
    from wtss.features import extract_view
    pl = placements(name, smoke)
    _c, cache, kind = cohort_info(name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = None
    fd = feat_dir(name, smoke, encoder)
    kmax = 2 if smoke else K_VERSIONS
    for k in range(kmax):
        ids = sorted(i for i, v in pl.items() if len(v) > k)
        for loc in ("in", "out"):
            f = fd / f"p{k}_{loc}.npz"
            if f.exists():
                continue
            if backend is None:
                backend = load_backend(encoder, dev)
            rend = make_paste_renderer(cache, kind, pl, k, loc, BACKEND_SIZE[encoder])
            extract_view(backend, ids, rend, f, dev, batch_size, workers, desc=f"{name}/p{k}_{loc}")
    C.log(cohort=name, extracted=str(fd))


def load_paste_features(name: str, smoke: bool, encoder: str = "dino518"):
    """{(k, loc): (X, {image_id: row})} and the placements (for donor lookup)."""
    fd = feat_dir(name, smoke, encoder)
    pl = placements(name, smoke)
    out = {}
    for f in sorted(fd.glob("p*_*.npz")):
        k, loc = f.stem[1:].split("_")
        z = np.load(f, allow_pickle=False)
        out[(int(k), loc)] = (z["X"], {s: j for j, s in enumerate(z["ids"].astype(str))})
    return out, pl


# ------------------------------------------------------------------------------------------------ probe (check)
MASK_FEATS = {"thyroid": "thyroid_all", "capsule": "capsule", "ovary": "ovary", "isic2019": "isic_all"}


def probe(name: str, smoke: bool, encoder: str = "dino518", n_boot: int = 2000) -> dict:
    """Registered check (prereg section 6): can a linear probe on frozen masked features tell a pasted in-ROI instance
    from a real in-ROI artifact? Positives: recipients with their version-0 instance pasted inside, masked. Negatives:
    images with a real artifact inside the ROI (present and overlap r >= 0.5), masked (existing cache). Split 80/20 by
    image group (stable hash); the larger class is subsampled to the smaller one in training; logistic regression,
    C = 1 (fixed). Reference probe (descriptive): recipients' plain masked view versus the same real-artifact images —
    how far the two image populations differ without any pasted instance."""
    from sklearn.linear_model import LogisticRegression
    from wtss import paths
    from wtss.utils import stable_int
    c, _cache, _kind = cohort_info(name)
    feats, _pl = load_paste_features(name, smoke, encoder)
    if (0, "in") not in feats:
        raise SystemExit(f"no pasted features for {name}; run --step extract")
    Xp, pos_p = feats[(0, "in")]
    z = np.load(paths.CACHE / "features" / MASK_FEATS[name] / BDIR[encoder] / "mask.npz", allow_pickle=False)
    pos_m = {s: j for j, s in enumerate(z["ids"].astype(str))}
    Xm = z["X"]
    grp = dict(zip(c.image_id, c.group))
    real = c[c.present & (c.r >= 0.5) & c.image_id.isin(pos_m)].image_id.tolist()
    pasted = [i for i in pos_p if i in grp]
    plain = [i for i in pasted if i in pos_m]
    if smoke:
        real = real[:60]
    res = {"cohort": name, "encoder": encoder}
    rng = np.random.default_rng(stable_int("r8_probe", name))

    def run(pos_ids, pos_X, tag):
        ids = pos_ids + real
        X = np.vstack([pos_X, Xm[[pos_m[i] for i in real]]])
        y = np.r_[np.ones(len(pos_ids)), np.zeros(len(real))].astype(int)
        test = np.array([stable_int("r8_probe_split", grp.get(i, i)) % 5 == 0 for i in ids])
        tr = np.flatnonzero(~test)
        n = min((y[tr] == 1).sum(), (y[tr] == 0).sum())
        if n < 5 or len(np.unique(y[test])) < 2:
            res[f"{tag}_auroc"] = float("nan")
            return
        tr = np.r_[rng.choice(tr[y[tr] == 1], n, replace=False), rng.choice(tr[y[tr] == 0], n, replace=False)]
        clf = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear").fit(X[tr], y[tr])
        p, yt = clf.predict_proba(X[test])[:, 1], y[test]
        auc = C.all_pairs_auc(yt, p)
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[b])) == 2:
                boots.append(C.all_pairs_auc(yt[b], p[b]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        res.update({f"{tag}_auroc": float(auc), f"{tag}_ci95_lo": float(lo), f"{tag}_ci95_hi": float(hi),
                    f"{tag}_n_test_pos": int((yt == 1).sum()), f"{tag}_n_test_neg": int((yt == 0).sum())})

    run(pasted, Xp[[pos_p[i] for i in pasted]], "pasted_vs_real")
    run(plain, Xm[[pos_m[i] for i in plain]], "reference_plain_vs_real")
    res["distinguishable"] = bool(res.get("pasted_vs_real_auroc", np.nan) > 0.75)
    out = C.stage_dir("checks", smoke)
    (out / f"locrand_probe_{name}.json").write_text(json.dumps(res, indent=1))
    C.log(**res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=COHORTS)
    ap.add_argument("--step", required=True, choices=("placements", "extract", "probe"))
    ap.add_argument("--encoder", default="dino518")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.step == "placements":
        s = placement_stats(a.cohort, a.smoke)
        out = C.stage_dir("checks", a.smoke)
        (out / f"locrand_placements_{a.cohort}.json").write_text(json.dumps(s, indent=1))
        C.log(**s)
    elif a.step == "extract":
        extract(a.cohort, a.smoke, a.encoder, a.batch_size, a.workers)
    else:
        probe(a.cohort, a.smoke, a.encoder, n_boot=200 if a.smoke else 2000)


if __name__ == "__main__":
    main()
