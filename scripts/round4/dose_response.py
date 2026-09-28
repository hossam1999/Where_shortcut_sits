"""R9 — dose-response of the mask gain over the real overlap r (docs/PREREGISTRATION_ROUND4.md).

  python scripts/round4/dose_response.py --cohort ovary --counts_only   # gate + merges -> counts.csv, bins_final.json
  python scripts/round4/dose_response.py --cohort ovary                 # full run + analysis
  python scripts/round4/dose_response.py --summary                      # Holm over cohorts, SUMMARY.{csv,md}, figure
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

BINS = [("b1", 0.0, 0.1), ("b2", 0.1, 0.3), ("b3", 0.3, 0.5), ("b4", 0.5, 0.75), ("b5", 0.75, 1.0 + 1e-9)]
MIN_CELL, MIN_REV_POS = 25, 40
N_BOOT = 10000


def add_bin(c, co, name, lo, hi):
    r = co["r"].fillna(-1.0)
    c[f"{name}_A1"] = co["present"] & (r >= lo) & (r < hi) & ~c.A0


def gate_one(c, co, name):
    from wtss.data.isic2019_spec import build_spec_envs, matched_pool
    mp = matched_pool(c, name)
    cells = mp.groupby(["a", "y"]).size()
    row = {"bin": name, **{f"A{x}_Y{y}": int(cells.get((x, y), 0)) for x in (0, 1) for y in (0, 1)}}
    a1 = mp[mp.a == 1].image_id
    rmap = dict(zip(c.image_id, co["r"]))
    row["x_mean_r"] = float(np.mean([rmap[i] for i in a1])) if len(a1) else float("nan")
    row["n_A1"] = int(len(a1))
    if min(row[k] for k in ("A0_Y0", "A0_Y1", "A1_Y0", "A1_Y1")) >= 2:
        envs = build_spec_envs(c, traps=(name,), seeds=(42,), group_col=co["group_col"])
        row["rev_pos_seed42"] = int(sum(envs[(name, 42, k, "test_rev")].y.sum() for k in range(5)))
    else:
        row["rev_pos_seed42"] = 0
    row["gate_ok"] = bool(min(row[k] for k in ("A0_Y0", "A0_Y1", "A1_Y0", "A1_Y1")) >= MIN_CELL and row["rev_pos_seed42"] >= MIN_REV_POS)
    return row


def counts(c, co, out: Path, smoke=False):
    """Apply the gate and the pre-registered merge rule; returns the final segments."""
    segs = [dict(name=n, lo=lo, hi=hi) for n, lo, hi in BINS]
    log, seen = [], {}
    while True:
        rows = []
        for s in segs:
            if s["name"] not in seen:
                add_bin(c, co, s["name"], s["lo"], s["hi"])
                seen[s["name"]] = gate_one(c, co, s["name"]) | {"lo": s["lo"], "hi": min(s["hi"], 1.0)}
            rows.append(seen[s["name"]])
        bad = [k for k, r in enumerate(rows) if not r["gate_ok"]]
        if not bad or len(segs) == 1 or smoke:
            break
        k = bad[0]
        mid = next(j for j, s in enumerate(segs) if s["lo"] <= 0.4 < s["hi"])
        if k < mid:
            j = k + 1
        elif k > mid:
            j = k - 1
        else:
            nb = [j for j in (k - 1, k + 1) if 0 <= j < len(segs)]
            j = min(nb, key=lambda j: rows[j]["n_A1"])
        a, b = sorted((k, j))
        m = dict(name=segs[a]["name"] + segs[b]["name"], lo=segs[a]["lo"], hi=segs[b]["hi"])
        log.append(f"{segs[k]['name']} failed the gate -> merged with {segs[j]['name']} into {m['name']}")
        segs = segs[:a] + [m] + segs[b + 1:]
    allrows = pd.DataFrame(list(seen.values()))
    allrows["final"] = allrows.bin.isin([s["name"] for s in segs])
    allrows.to_csv(out / "counts.csv", index=False)
    fin = [dict(s, x=seen[s["name"]]["x_mean_r"], gate_ok=seen[s["name"]]["gate_ok"]) for s in segs]
    status = "tested" if len(fin) >= 3 and all(s["gate_ok"] for s in fin) else "descriptive"
    js = {"bins": fin, "merges": log, "status": status, "min_cell": MIN_CELL, "min_rev_pos": MIN_REV_POS}
    (out / "bins_final.json").write_text(json.dumps(js, indent=1))
    print(allrows.to_string(), "\n", json.dumps(js, indent=1), flush=True)
    return js


def _boot(job):
    kind, args, seed = job
    from wtss import stats_crossed as X
    if kind == "pair":
        return X.hierarchical_paired_bootstrap(*args, N_BOOT, seed)
    frames, w = args
    terms = []
    for P, wb in zip(frames, w):
        terms += X._pair_terms(P, "mask", "erm", "test_rev", "seed", coef=float(wb))
    point, arr = X.replicates(terms, N_BOOT, seed)
    return X._summary(point, arr, {"estimand": "OLS slope of rev(mask)-rev(ERM) on bin mean r"})


def analyse(out: Path, cohort: str, js: dict):
    from scipy.stats import spearmanr
    P = pd.read_csv(out / "predictions.csv.gz")
    P = P[P.env == "test_rev"][["trap", "seed", "method", "env", "image_id", "y", "prob"]]
    names = [b["name"] for b in js["bins"]]
    x = np.array([b["x"] for b in js["bins"]], float)
    frames = [P[P.trap == n] for n in names]
    jobs = [("pair", (f, "mask", "erm", "test_rev"), 20261301 + k) for k, f in enumerate(frames)]
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum() if len(x) >= 2 else None
    if w is not None:
        jobs.append(("slope", (frames, w), 20261399))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_boot, jobs))
    per = pd.DataFrame([{"cohort": cohort, "bin": n, "x_mean_r": xx, "gain": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"],
                         "ci95_hi": r["ci95_hi"], "boot_seed": j[2]} for n, xx, r, j in zip(names, x, res, jobs)])
    per.to_csv(out / "bin_gains.csv", index=False)
    g = per.gain.to_numpy()
    zero = None
    for k in range(len(g) - 1):
        if np.sign(g[k]) != np.sign(g[k + 1]):
            zero = float(x[k] + (x[k + 1] - x[k]) * g[k] / (g[k] - g[k + 1]))
            break
    sl = res[-1] if w is not None else {}
    row = {"cohort": cohort, "status": js["status"], "n_bins": len(names), "bins": ", ".join(names),
           "slope": sl.get("seed_delta_mean"), "ci95_lo": sl.get("ci95_lo"), "ci95_hi": sl.get("ci95_hi"),
           "p_boot_two_sided": sl.get("p_boot_two_sided"), "boot_seed": 20261399,
           "spearman_x_gain": float(spearmanr(x, g)[0]) if len(x) >= 3 else float("nan"), "zero_crossing_r": zero,
           "estimator": sl.get("estimator", "")}
    pd.DataFrame([row]).to_csv(out / "slope.csv", index=False)
    print(per.round(3).to_string(), "\n", pd.Series(row).to_string(), flush=True)


def summary(root: Path):
    sl = [pd.read_csv(root / c / "slope.csv") for c in K.COHORTS if (root / c / "slope.csv").exists()]
    if not sl:
        raise SystemExit("no cohort finished yet")
    d = pd.concat(sl, ignore_index=True)
    t = d.status == "tested"
    d["p_holm"] = np.nan
    if t.any():
        d.loc[t, "p_holm"] = K.holm(d.loc[t, "p_boot_two_sided"].to_numpy())
    d["H9_supported"] = t & (d.slope < 0) & (d.p_holm < 0.05)
    d.to_csv(root / "SUMMARY.csv", index=False)
    per = pd.concat([pd.read_csv(root / c / "bin_gains.csv") for c in d.cohort], ignore_index=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(d), figsize=(3.2 * len(d), 2.8), squeeze=False)
    for a, (c, q) in zip(ax[0], per.groupby("cohort", sort=False)):
        a.errorbar(q.x_mean_r, q.gain, yerr=[q.gain - q.ci95_lo, q.ci95_hi - q.gain], fmt="o-", capsize=3)
        a.axhline(0, color="k", lw=0.6)
        a.set_title(K.LABEL[c], fontsize=9)
        a.set_xlabel("mean overlap r of the bin")
    ax[0][0].set_ylabel("rev AUROC: mask - ERM")
    fig.tight_layout()
    fig.savefig(root / "dose_response.pdf")
    L = ["# R9 — dose-response over the real overlap r (DINOv2 ViT-B/14 @518)", "",
         f"Environment: {json.dumps(K.env_info())}", "",
         "Slope of g_b = rev(mask) - rev(ERM) on the bin mean r; crossed bootstrap; Holm over tested cohorts. H9: slope < 0.", ""]
    dd = d.assign(cohort=d.cohort.map(K.LABEL), slope_ci=[f"{r['slope']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"
                                                          if pd.notna(r["slope"]) else "" for r in d.to_dict("records")])
    L += [K.md_table(dd[["cohort", "status", "bins", "slope_ci", "p_holm", "H9_supported", "spearman_x_gain", "zero_crossing_r"]]), ""]
    pp = per.assign(cohort=per.cohort.map(K.LABEL), gain_ci=[f"{r['gain']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]" for r in per.to_dict("records")])
    L += ["Per-bin gains:", "", K.md_table(pp[["cohort", "bin", "x_mean_r", "gain_ci"]]), ""]
    for c in d.cohort:
        js = json.loads((root / c / "bins_final.json").read_text())
        L.append(f"- {K.LABEL[c]}: merges: {'; '.join(js['merges']) or 'none'}")
    (root / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=K.COHORTS)
    ap.add_argument("--counts_only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_jobs", type=int, default=4)
    a = ap.parse_args()
    root = K.OUT / ("_smoke" if a.smoke else "") / "dose_response"
    if a.summary:
        return summary(root)
    co = K.trap_cohort(a.cohort)
    c = co["c"]
    if a.smoke:
        for n, lo, hi in BINS:
            add_bin(c, co, n, lo, hi)
        c = K.smoke_subset(c, [n for n, _, _ in BINS], n=60)
        co = _rebind(co, c)
    out = root / a.cohort
    out.mkdir(parents=True, exist_ok=True)
    f = out / "bins_final.json"
    js = json.loads(f.read_text()) if f.exists() else counts(c, co, out, a.smoke)
    if a.counts_only:
        return
    import torch
    from wtss.data.isic2019_spec import build_spec_envs
    from wtss.experiments.real_traps import make_renderers
    from wtss.experiments.spec_traps import run_spec
    names = [b["name"] for b in js["bins"]]
    for b in js["bins"]:
        add_bin(c, co, b["name"], b["lo"], b["hi"] if b["hi"] < 1 else 1.0 + 1e-9)
    envs = build_spec_envs(c, traps=names, group_col=co["group_col"])
    if a.smoke:
        envs = K.smoke_envs(envs)
    dev = torch.device(a.device)
    pool = K.pool_of(envs, names)
    rend = make_renderers(co["cache"], 518, co["donors"])
    fdir = K.FEAT / ("_smoke" if a.smoke else "") / "dose_response" / a.cohort / K.BDIR
    olds = co["old"] + [K.FEAT / "masking_variants" / a.cohort / K.BDIR]
    srcs = {v: [(d / f"{v}.npz", lambda i: i) for d in olds] for v in ("erm", "mask")}
    if not (out / "predictions.csv.gz").exists():
        K.assemble_views(pool, ("erm", "mask"), rend, fdir, srcs, dev, workers=a.workers)
        run_spec(envs, co["cache"], K.BACKBONE, out, fdir, co["donors"], arms=("erm", "mask"), traps=tuple(names),
                 device=dev, workers=a.workers, n_jobs=a.n_jobs, folds=[0] if a.smoke else range(5))
    analyse(out, a.cohort, js)


def _rebind(co, c):
    """Cohort dict for a subset table c (smoke runs): presence and overlap aligned to c's rows."""
    sel = lambda s: pd.Series(s.to_numpy(), index=co["c"].image_id).loc[c.image_id].reset_index(drop=True)
    return dict(co, c=c, present=sel(co["present"]), r=sel(co["r"]))


if __name__ == "__main__":
    main()
