"""Round 7 — operating points with thresholds matched on the test set (docs/PREREGISTRATION_ROUND7.md).

Each arm gets its own threshold, chosen on the test images so that both arms reach the same test specificity (or
sensitivity); sensitivity is then compared in the conflicting subgroup (malignant with an in-ROI caliper for thyroid).
Only saved predictions are read.
  python scripts/round7/matched_thresholds.py [--smoke]
Outputs: results/round7/{matched_thresholds.csv, tm.json, SUMMARY.md} (smoke: results/round7/_smoke/).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths

ROOT = paths.REPO_ROOT
MATCH = {"M-spec80": ("spec", 0.80), "M-spec90": ("spec", 0.90), "M-sens80": ("sens", 0.80)}
KEYS = ("sens", "sens_conflict", "sens_aligned", "spec")
N_BOOT, SEED = 10000, 20260928
FT_PRED = ROOT / "results" / "round6" / "ft_natural" / "predictions.csv.gz"
OTHER = ("isic_BCN", "isic_HAM", "isic_MSK", "capsule")


def _script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OP = _script("op_round7", "scripts/analysis/operating_points.py")


def arrays(p: pd.DataFrame, hard_pos_a: int):
    """Per seed: test labels, conflicting-positive flag, ERM and mask scores, image index (shared bootstrap weights)."""
    te = p[(p.env == "clean") & p.method.isin(["erm", "mask"])].copy()
    te["image_id"] = te.image_id.astype(str)
    ids = sorted(te.image_id.unique())
    pos = {k: j for j, k in enumerate(ids)}
    S = {}
    for s, q in te.groupby("seed"):
        e = q[q.method == "erm"].sort_values("image_id")
        m = q[q.method == "mask"].sort_values("image_id")
        if not (e.image_id.to_numpy() == m.image_id.to_numpy()).all():
            raise SystemExit(f"seed {s}: ERM and mask were scored on different test images")
        y, a = e.y.to_numpy().astype(int), e.artifact_present.to_numpy().astype(int)
        S[s] = {"y": y, "conf": (y == 1) & (a == hard_pos_a), "erm": e.prob.to_numpy(), "mask": m.prob.to_numpy(),
                "i": np.array([pos[i] for i in e.image_id])}
    return S, len(ids)


def rates(y, prob, conf, w, kind, target) -> dict:
    th = OP.threshold_w(y, prob, w, kind, target)
    pr = prob >= th
    pos = y == 1
    share = lambda sel: float(np.sum(w * pr * sel) / max(np.sum(w * sel), 1e-9))
    return {"sens": share(pos), "sens_conflict": share(conf), "sens_aligned": share(pos & ~conf),
            "spec": float(np.sum(w * ~pr * (y == 0)) / max(np.sum(w * (y == 0)), 1e-9))}


def _deltas(S, seeds, W):
    out = {m: {k: [] for k in KEYS} for m in MATCH}
    for s in seeds:
        d = S[s]
        w = np.ones(len(d["y"])) if W is None else W[d["i"]]
        for m, (kind, tg) in MATCH.items():
            ra = rates(d["y"], d["mask"], d["conf"], w, kind, tg)
            rr = rates(d["y"], d["erm"], d["conf"], w, kind, tg)
            for k in KEYS:
                out[m][k].append(ra[k] - rr[k])
    return {m: {k: float(np.mean(v)) for k, v in dd.items()} for m, dd in out.items()}


def analyse(job):
    model, cohort, p, hard_pos_a, n_boot = job
    S, n_ids = arrays(p, hard_pos_a)
    seeds = sorted(S)
    # point estimates (unit weights), per arm and as mask - ERM
    level = {m: {arm: {k: [] for k in KEYS} for arm in ("erm", "mask")} for m in MATCH}
    for s in seeds:
        d = S[s]
        for m, (kind, tg) in MATCH.items():
            for arm in ("erm", "mask"):
                r = rates(d["y"], d[arm], d["conf"], np.ones(len(d["y"])), kind, tg)
                for k in KEYS:
                    level[m][arm][k].append(r[k])
    rng = np.random.default_rng(SEED)
    reps = {m: {k: [] for k in KEYS} for m in MATCH}
    for _ in range(n_boot):
        W = rng.poisson(1.0, n_ids).astype(float)
        pick = rng.choice(seeds, len(seeds), replace=True)
        dd = _deltas(S, pick, W)
        for m in MATCH:
            for k in KEYS:
                reps[m][k].append(dd[m][k])
    rows = []
    n_conf = int(S[seeds[0]]["conf"].sum())
    for m in MATCH:
        for k in KEYS:
            arr = np.asarray(reps[m][k], float)
            arr = arr[np.isfinite(arr)]
            e, a = float(np.mean(level[m]["erm"][k])), float(np.mean(level[m]["mask"][k]))
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rows.append({"model": model, "cohort": cohort, "match": m, "metric": k, "erm_value": e, "mask_value": a,
                         "delta": a - e, "ci95_lo": float(lo), "ci95_hi": float(hi),
                         "p_less": float(min(1.0, max((arr >= 0).mean(), 1.0 / len(arr)))),
                         "n_conflicting_positives": n_conf, "n_seeds": len(seeds), "n_boot": int(len(arr)), "boot_seed": SEED})
    return rows


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[j]))
        adj[j] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "round7" / ("_smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    n_boot = 200 if a.smoke else N_BOOT
    if not FT_PRED.exists():
        raise SystemExit(f"missing {FT_PRED} (round 6, B2)")
    ft = pd.read_csv(FT_PRED)
    ft = ft[ft.env.isin(["clean", "val_groups"])]
    frozen, hpa = OP.load("thyroid")
    te = ft[(ft.env == "clean") & (ft.method == "erm")]
    ft_hpa = 1 if te[te.y == 1].artifact_present.mean() < te[te.y == 0].artifact_present.mean() else 0
    if ft_hpa != 1 or hpa != 1:
        raise SystemExit(f"conflicting direction differs from the registration (fine-tuned {ft_hpa}, frozen {hpa})")
    if not a.smoke:  # the registered FT3 subgroup (Stage 5, round 6)
        for name, q in (("fine-tuned", ft), ("frozen", frozen)):
            t = q[(q.env == "clean") & (q.method == "erm")]
            n = t[(t.y == 1) & (t.artifact_present == 1)].groupby("seed").size().unique()
            if list(n) != [78]:
                raise SystemExit(f"{name} predictions: conflicting subgroup sizes {list(n)}, registered 78")
    jobs = [("finetuned_round6", "thyroid", ft, 1, n_boot), ("frozen_stage5", "thyroid", frozen, hpa, n_boot)]
    for c in (OTHER[-1:] if a.smoke else OTHER):
        p, h = OP.load(c)
        jobs.append(("frozen_stage5", c, p, h, n_boot))
    with ProcessPoolExecutor(max_workers=min(len(jobs), 8), mp_context=mp.get_context("fork")) as ex:
        rows = [r for rs in ex.map(analyse, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out / "matched_thresholds.csv", index=False)
    pick = lambda mo: df[(df.model == mo) & (df.cohort == "thyroid") & (df.match == "M-spec80") &
                         (df.metric == "sens_conflict")].iloc[0]
    tm = {"TM1": pick("finetuned_round6"), "TM2": pick("frozen_stage5")}
    adj = holm(np.array([tm["TM1"].p_less, tm["TM2"].p_less]))
    rec = {}
    for (k, r), ph in zip(tm.items(), adj):
        ok = r.delta < 0 and r.ci95_hi < 0 and ph < 0.05
        rec[k] = {"model": r.model, "estimate": r.delta, "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi, "p": r.p_less,
                  "p_holm": float(ph), "erm_value": r.erm_value, "mask_value": r.mask_value,
                  "n_conflicting_positives": int(r.n_conflicting_positives),
                  "verdict": "SUPPORTED" if ok else "NOT SUPPORTED"}
    (out / "tm.json").write_text(json.dumps(rec, indent=2))
    lines = ["# Round 7 — thresholds matched on the test set", "",
             "| hypothesis | model | conflicting positives | sensitivity ERM → mask | mask − ERM [95% CI] | p | Holm p | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in rec.items():
        lines.append(f"| {k} | {r['model']} | {r['n_conflicting_positives']} | {r['erm_value']:.3f} → {r['mask_value']:.3f} | "
                     f"{r['estimate']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | {r['p']:.4f} | {r['p_holm']:.4f} | {r['verdict']} |")
    lines += ["", "## All matched thresholds (descriptive except TM1–TM2)", "",
              "| model | cohort | match | metric | ERM | mask | mask − ERM [95% CI] |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.model} | {r.cohort} | {r.match} | {r.metric} | {r.erm_value:.3f} | {r.mask_value:.3f} | "
                     f"{r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
