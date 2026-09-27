"""Operating points on the unaltered test distributions (docs/PREREGISTRATION_REVIEW2.md, R3).

Thresholds are fixed on each seed's validation predictions (never on test): OP1 max balanced accuracy, OP2/OP3
validation specificity >= 0.80/0.90, OP4/OP5 validation sensitivity >= 0.80/0.90. Test sensitivity/specificity
overall and in the shortcut-conflicting subgroups. Hierarchical paired bootstrap: seeds resampled, then validation
and test images within each seed (thresholds re-estimated on each resampled validation set), arm − reference per seed.

  python scripts/analysis/operating_points.py            -> results/review2/operating_points*.csv
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from wtss import paths

COHORTS = ["thyroid", "isic_BCN", "isic_HAM", "isic_MSK", "capsule"]
OPS = {"OP1_maxBA": ("ba", None), "OP2_spec0.80": ("spec", 0.80), "OP3_spec0.90": ("spec", 0.90),
       "OP4_sens0.80": ("sens", 0.80), "OP5_sens0.90": ("sens", 0.90)}
PAIRS = [("mask", "erm"), ("mte_balanced", "mask"), ("balanced", "mask")]
N_BOOT = 10000


def threshold(y: np.ndarray, p: np.ndarray, kind: str, target) -> float:
    """Predict positive iff p >= t. ba: maximise (sens + spec)/2; spec: max sensitivity s.t. spec >= target;
    sens: max specificity s.t. sens >= target."""
    t = np.unique(p)[::-1]  # candidate thresholds, descending
    pos, neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    last = np.searchsorted(-ps, -t, side="right") - 1  # index of the last score >= t
    sens, spec = tp[last] / pos, 1 - fp[last] / neg
    if kind == "ba":
        j = int(np.argmax(sens + spec))
    elif kind == "spec":
        ok = np.flatnonzero(spec >= target)
        j = int(ok[np.argmax(sens[ok])]) if len(ok) else 0
    else:
        ok = np.flatnonzero(sens >= target)
        j = int(ok[np.argmax(spec[ok])]) if len(ok) else len(t) - 1
    return float(t[j])


def rates(y, p, a, thr, hard_pos_a):
    pred = p >= thr
    r = {"sens": pred[y == 1].mean(), "spec": (~pred[y == 0]).mean()}
    hp, hn = (y == 1) & (a == hard_pos_a), (y == 0) & (a == 1 - hard_pos_a)
    r["sens_conflict"] = pred[hp].mean() if hp.any() else np.nan
    r["spec_conflict"] = (~pred[hn]).mean() if hn.any() else np.nan
    return r


def load(cohort: str):
    d = paths.RESULTS / "natural" / f"{cohort}_dino518_repro"
    p = pd.read_csv(d / "predictions.csv.gz")
    p = p[p.env.isin(["clean", "val_groups"])].copy()
    te = p[p.env == "clean"]
    # a = in-ROI artifact (same definition as run_natural); hard direction from the test-set association
    pa1, pa0 = te[te.y == 1].artifact_present.mean(), te[te.y == 0].artifact_present.mean()
    hard_pos_a = 1 if pa1 < pa0 else 0  # malignant cases WITH the artifact conflict when it marks benign
    return p, hard_pos_a


def per_seed(p: pd.DataFrame, hard_pos_a: int) -> pd.DataFrame:
    rows = []
    for (m, s), q in p.groupby(["method", "seed"]):
        v, t = q[q.env == "val_groups"], q[q.env == "clean"]
        for op, (kind, tg) in OPS.items():
            thr = threshold(v.y.to_numpy(), v.prob.to_numpy(), kind, tg)
            rows.append({"method": m, "seed": s, "op": op, "thr": thr,
                         **rates(t.y.to_numpy(), t.prob.to_numpy(), t.artifact_present.to_numpy(), thr, hard_pos_a)})
    return pd.DataFrame(rows)


def _boot(args):
    arrays, arm, ref, hard_pos_a, seed = args
    rng = np.random.default_rng(seed)
    seeds = sorted(arrays)
    out = {op: {k: np.empty(N_BOOT) for k in ("sens", "spec", "sens_conflict", "spec_conflict")} for op in OPS}
    for b in range(N_BOOT):
        pick = rng.choice(seeds, len(seeds), replace=True)
        acc = {op: {k: [] for k in ("sens", "spec", "sens_conflict", "spec_conflict")} for op in OPS}
        for s in pick:
            vy, vp_a, vp_r, ty, tp_a, tp_r, ta = arrays[s]
            iv = rng.integers(0, len(vy), len(vy))
            it = rng.integers(0, len(ty), len(ty))
            for op, (kind, tg) in OPS.items():
                ta_ = threshold(vy[iv], vp_a[iv], kind, tg)
                tr_ = threshold(vy[iv], vp_r[iv], kind, tg)
                ra = rates(ty[it], tp_a[it], ta[it], ta_, hard_pos_a)
                rr = rates(ty[it], tp_r[it], ta[it], tr_, hard_pos_a)
                for k in ra:
                    acc[op][k].append(ra[k] - rr[k])
        for op in OPS:
            for k, v in acc[op].items():
                out[op][k][b] = np.nanmean(v)
    return out


def main():
    rows, seeds_all = [], []
    for cohort in COHORTS:
        try:
            p, hpa = load(cohort)
        except FileNotFoundError:
            print(f"[op] {cohort}: no repro predictions yet", flush=True)
            continue
        ps = per_seed(p, hpa)
        ps.insert(0, "cohort", cohort)
        seeds_all.append(ps)
        jobs = []
        for arm, ref in PAIRS:
            if not {arm, ref} <= set(p.method):
                continue
            arrays = {}
            for s, q in p.groupby("seed"):
                g = lambda m, e: q[(q.method == m) & (q.env == e)].sort_values("image_id")
                va, vr, tea, ter = g(arm, "val_groups"), g(ref, "val_groups"), g(arm, "clean"), g(ref, "clean")
                assert (va.image_id.values == vr.image_id.values).all() and (tea.image_id.values == ter.image_id.values).all()
                arrays[int(s)] = (va.y.to_numpy(), va.prob.to_numpy(), vr.prob.to_numpy(), tea.y.to_numpy(),
                                  tea.prob.to_numpy(), ter.prob.to_numpy(), tea.artifact_present.to_numpy())
            jobs.append((arrays, arm, ref, hpa, 20260928 + sum(map(ord, cohort + arm + ref))))
        with ProcessPoolExecutor(len(jobs)) as ex:
            res = list(ex.map(_boot, jobs))
        for (_, arm, ref, _, _), r in zip(jobs, res):
            for op in OPS:
                for k, arr in r[op].items():
                    pa = ps[(ps.method == arm) & (ps.op == op)][k].mean()
                    pr = ps[(ps.method == ref) & (ps.op == op)][k].mean()
                    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
                    rows.append({"cohort": cohort, "op": op, "metric": k, "arm": arm, "ref": ref, "arm_value": pa,
                                 "ref_value": pr, "delta": pa - pr, "ci95_lo": lo, "ci95_hi": hi,
                                 "ci_excludes_zero": bool(lo > 0 or hi < 0)})
        print(f"[op] {cohort} done", flush=True)
    out = paths.ensure(paths.RESULTS / "review2")
    pd.concat(seeds_all).to_csv(out / "operating_points_per_seed.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(out / "operating_points.csv", index=False)
    th = d[(d.cohort == "thyroid") & (d.arm == "mask") & (d.metric.isin(["sens", "sens_conflict"]))]
    s1 = th[(th.op == "OP1_maxBA") & (th.metric == "sens")]
    others = th[(th.op != "OP1_maxBA") & (th.metric == "sens")]
    verdict = {"S1_OP1_sens_loss": bool(len(s1) and s1.ci95_hi.iloc[0] < 0),
               "S2_ops_with_sens_loss": int((others.ci95_hi < 0).sum()),
               "S3_conflict_loss_OP2_OP3": th[(th.op.isin(["OP2_spec0.80", "OP3_spec0.90"])) & (th.metric == "sens_conflict")]
               [["op", "delta", "ci95_lo", "ci95_hi"]].round(3).to_dict("records")}
    verdict["may_say_masking_lowers_sensitivity"] = verdict["S1_OP1_sens_loss"] and verdict["S2_ops_with_sens_loss"] >= 3
    (out / "operating_points_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
