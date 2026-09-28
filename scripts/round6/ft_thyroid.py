"""Part B — validation-only recipe, then the natural split and the thyroid traps.

  python scripts/round6/ft_thyroid.py --select
  python scripts/round6/ft_thyroid.py --natural
  python scripts/round6/ft_thyroid.py --traps
  python scripts/round6/ft_thyroid.py --analyse
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from wtss.experiments.finetune import train_eval  # noqa: E402
from wtss.experiments.real_traps import make_renderers  # noqa: E402
from wtss.utils import stable_int  # noqa: E402

ARCHS = ("convnext_tiny.fb_in22k_ft_in1k", "resnet50", "vit_base_patch14_dinov2.lvd142m")
LRS = (1e-4, 3e-5)
EPOCHS = (8, 20)
SEEDS = (42, 123, 456, 789, 2026)
ARMS = ("erm", "mask", "balanced", "mask_balanced")
BENCH = 0.773


def _natural():
    p = C.ROOT / "scripts" / "run_natural.py"
    spec = importlib.util.spec_from_file_location("run_natural_r6", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load("thyroid")


def model_kwargs(arch: str):
    if arch == "vit_base_patch14_dinov2.lvd142m":
        return {"img_size": 224}
    return None


def parts(c, seed: int, smoke: bool):
    tv, te = c[c.split == "trainval"].copy(), c[c.split == "test"].copy()
    if smoke:
        rng = np.random.default_rng(seed)
        tv = tv.iloc[rng.choice(len(tv), 80, replace=False)]
        te = te.iloc[rng.choice(len(te), 40, replace=False)]
    val_g = {g for g in tv.group.unique() if stable_int("natural_val", seed, g) % 5 == 0}
    if smoke and not val_g:
        val_g = set(list(tv.group.unique())[:1])
    va, tr = tv[tv.group.isin(val_g)], tv[~tv.group.isin(val_g)]
    cols = ["image_id", "y", "a"]
    return tr[cols].reset_index(drop=True), va[cols].reset_index(drop=True), te[cols].reset_index(drop=True)


def _auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def select(smoke: bool):
    out = C.out_root(smoke)
    path = out / "ft_recipe.json"
    done = json.loads(path.read_text()) if path.exists() else {"runs": [], "choice": None}
    have = {(r["arch"], r["lr"], r["epochs"]) for r in done["runs"]}
    c, cache, _, _ = _natural()
    tr, va, _te = parts(c, 42, smoke)
    render = make_renderers(cache, 518, [])
    grid = [(ARCHS[1], LRS[0], 1)] if smoke else [(a, lr, ep) for a in ARCHS for lr in LRS for ep in EPOCHS]
    E = {"train_corr": tr, "clean_val": va, "clean_test": va, "test_corr": va, "test_rev": va}
    for arch, lr, ep in grid:
        key = (arch, float(lr), int(ep))
        if key in have and not smoke:
            continue
        C.log(select=arch, lr=lr, epochs=ep, n_train=len(tr), n_val=len(va))
        res = train_eval("erm", E, render, arch=arch, epochs=int(ep), lr=float(lr), seed=42,
                         device=torch.device("cuda"), workers=6, model_kwargs=model_kwargs(arch))
        prob = res["clean_test"][0].predict_proba(None)[:, 1]
        row = {"arch": arch, "lr": float(lr), "epochs": int(ep), "val_auroc": _auc(va.y, prob), "seed": 42, "arm": "erm"}
        done["runs"] = [r for r in done["runs"] if (r["arch"], r["lr"], r["epochs"]) != key] + [row]
        done["runs"] = sorted(done["runs"], key=lambda r: (ARCHS.index(r["arch"]) if r["arch"] in ARCHS else 99, r["lr"], r["epochs"]))
        path.write_text(json.dumps(done, indent=2))
        C.log(val_auroc=row["val_auroc"])
    if len(done["runs"]) == len(grid):
        best = max(done["runs"], key=lambda r: (r["val_auroc"], -ARCHS.index(r["arch"]) if r["arch"] in ARCHS else 0, -r["epochs"]))
        # tie: highest validation AUROC, then the earlier architecture in the registered grid, then the smaller epoch count
        ordered = sorted(done["runs"], key=lambda r: (-r["val_auroc"], ARCHS.index(r["arch"]) if r["arch"] in ARCHS else 99, r["lr"], r["epochs"]))
        best = ordered[0]
        done["choice"] = {k: best[k] for k in ("arch", "lr", "epochs", "val_auroc")}
        done["tie_break"] = "highest validation AUROC; ties keep the earlier registered grid order (architecture, then learning rate, then epochs)"
        path.write_text(json.dumps(done, indent=2))
    print(path.read_text())


def _recipe(smoke: bool) -> dict:
    p = C.out_root(smoke) / "ft_recipe.json"
    d = json.loads(p.read_text())
    if not d.get("choice"):
        raise SystemExit(f"no recipe choice in {p}; run --select and commit it before the final models")
    return d["choice"]


def natural(smoke: bool):
    choice = _recipe(smoke)
    c, cache, _, _ = _natural()
    render = make_renderers(cache, 518, [])
    dest = C.out_root(smoke) / "ft_natural"
    dest.mkdir(parents=True, exist_ok=True)
    pf = dest / "predictions.csv.gz"
    done = pd.read_csv(pf) if pf.exists() else None
    frames = [] if done is None else [done]
    seeds = (42,) if smoke else SEEDS
    arms = ("erm",) if smoke else ARMS
    epochs = 1 if smoke else int(choice["epochs"])
    for seed in seeds:
        tr, va, te = parts(c, seed, smoke)
        E = {"train_corr": tr, "clean_val": va, "clean_test": te, "test_corr": va, "test_rev": te}
        for arm in arms:
            if done is not None and ((done.seed == seed) & (done.method == arm)).any():
                continue
            C.log(natural=arm, seed=seed)
            res = train_eval(arm, E, render, arch=choice["arch"], epochs=epochs, lr=float(choice["lr"]), seed=seed,
                             device=torch.device("cuda"), workers=6, model_kwargs=model_kwargs(choice["arch"]))
            from wtss.evaluation import evaluate
            for env, (clf, thr, d) in res.items():
                if env == "test_rev":
                    continue  # same images as clean_test; the renamed envs are the ones the analysis reads
                name = {"clean_test": "clean", "test_corr": "val_groups"}[env]
                _r, f = evaluate(clf, thr, None, d.y.to_numpy(), d.image_id.to_numpy(), d.a.to_numpy(),
                                 {"method": arm, "env": name, "seed": seed})
                frames.append(f)
            pd.concat(frames, ignore_index=True).to_csv(pf, index=False, compression="gzip")
    C.log(natural="predictions", path=str(pf))


def traps(smoke: bool):
    choice = _recipe(smoke)
    epochs = "1" if smoke else str(int(choice["epochs"]))
    folds = ["0"] if smoke else ["0", "1", "2", "3", "4"]
    tag = "round6_smoke" if smoke else "round6"
    seeds = ("42",) if smoke else ("42", "123")
    for s in seeds:
        cmd = [sys.executable, str(C.ROOT / "scripts" / "run_finetune_spec.py"),
               "--cohort", "thyroid", "--arch", choice["arch"], "--lr", str(choice["lr"]),
               "--epochs", epochs, "--arms", "erm", "mask", "--env_seed", s, "--tag", tag, "--folds", *folds]
        if model_kwargs(choice["arch"]):
            cmd += ["--img_size", str(model_kwargs(choice["arch"])["img_size"])]
        C.log(traps=" ".join(cmd))
        subprocess.check_call(cmd, cwd=str(C.ROOT))


def _nisic():
    """Load the round-4 natural script. Its `import common` must see round-4 common, not this package."""
    if getattr(_nisic, "mod", None) is not None:
        return _nisic.mod
    saved = sys.modules.get("common")
    sys.modules["common"] = C.R4
    sys.path.insert(0, str(C.ROOT / "scripts" / "round4"))
    path = C.ROOT / "scripts" / "round4" / "natural_isic2020.py"
    spec = importlib.util.spec_from_file_location("natural_isic2020_r6", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if saved is not None:
        sys.modules["common"] = saved
    _nisic.mod = mod
    return mod


def _op_replicates(p: pd.DataFrame, n_boot: int, seed: int = 20260928):
    """Same crossed operating-point bootstrap as natural_isic2020.op_crossed, returning replicate deltas."""
    N = _nisic()
    op = N.K.load_script("op", "scripts/analysis/operating_points.py")
    ids = sorted(p.image_id.astype(str).unique())
    pos = {k: j for j, k in enumerate(ids)}
    S = {}
    for s, q in p.groupby("seed"):
        g = lambda m, e: q[(q.method == m) & (q.env == e)].sort_values("image_id")
        va, vr, ta, tr = g("mask", "val_groups"), g("erm", "val_groups"), g("mask", "clean"), g("erm", "clean")
        S[s] = dict(vy=va.y.to_numpy(), vpa=va.prob.to_numpy(), vpr=vr.prob.to_numpy(),
                    vi=np.array([pos[i] for i in va.image_id.astype(str)]),
                    ty=ta.y.to_numpy(), tpa=ta.prob.to_numpy(), tpr=tr.prob.to_numpy(),
                    ti=np.array([pos[i] for i in ta.image_id.astype(str)]),
                    ta=ta.artifact_present.to_numpy())
    seeds = sorted(S)
    rng = np.random.default_rng(seed)
    keys = ("sens", "sens_conflict")
    res = {o: {k: [] for k in keys} for o in op.OPS}
    for _ in range(n_boot):
        Wt = rng.poisson(1.0, len(ids)).astype(float)
        pick = rng.choice(seeds, len(seeds), replace=True)
        acc = {o: {k: [] for k in keys} for o in op.OPS}
        for s in pick:
            d = S[s]
            wv, wt = Wt[d["vi"]], Wt[d["ti"]]
            y, t = d["ty"], d["ta"]
            hp = (y == 1) & (t == 1)
            for o, (kind, tg) in op.OPS.items():
                vals = []
                for vp, tp in ((d["vpa"], d["tpa"]), (d["vpr"], d["tpr"])):
                    th = op.threshold_w(d["vy"], vp, wv, kind, tg)
                    pr = tp >= th
                    vals.append({"sens": np.sum(wt * pr * (y == 1)) / max(np.sum(wt * (y == 1)), 1e-9),
                                 "sens_conflict": np.sum(wt * pr * hp) / max(np.sum(wt * hp), 1e-9)})
                for k in keys:
                    acc[o][k].append(vals[0][k] - vals[1][k])
        for o in op.OPS:
            for k in keys:
                res[o][k].append(float(np.mean(acc[o][k])))
    return {o: {k: np.asarray(v, float) for k, v in d.items()} for o, d in res.items()}


def analyse(smoke: bool):
    from wtss import stats_crossed as X
    out = C.out_root(smoke)
    choice = json.loads((out / "ft_recipe.json").read_text()).get("choice")
    pf = out / "ft_natural" / "predictions.csv.gz"
    if not pf.exists():
        raise SystemExit(f"missing {pf}")
    p = pd.read_csv(pf)
    te = p[p.env == "clean"]
    erm = te[te.method == "erm"]
    aucs = [float(roc_auc_score(q.y, q.prob)) for _, q in erm.groupby("seed") if q.y.nunique() == 2]
    ft0 = {"id": "FT0", "estimate": float(np.mean(aucs)) if aucs else float("nan"), "benchmark": BENCH,
           "verdict": "benchmark-level" if aucs and float(np.mean(aucs)) >= BENCH else "below benchmark",
           "p": None, "note": "descriptive; not in the Holm family"}
    hard = ((te.y == 1) & (te.artifact_present == 1)) | ((te.y == 0) & (te.artifact_present == 0))
    q = te[hard & te.method.isin(["mask", "erm"])]
    point, arr = X.replicates(X._pair_terms(q, "mask", "erm", "clean", "seed"), 2000 if smoke else 10000, 20260928)
    est = float(np.mean(list(point.values())))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    ft1 = {"id": "FT1", "estimate": est, "ci95_lo": float(lo), "ci95_hi": float(hi),
           "p": C.one_sided_p(arr, "less"), "direction": "mask-erm hard AUROC < 0"}
    op_rows = []
    ft2 = {"id": "FT2", "note": "mask and erm predictions with both envs are required"}
    ft3 = {"id": "FT3"}
    if {"mask", "erm"} <= set(p.method) and {"clean", "val_groups"} <= set(p.env):
        N = _nisic()
        op_rows = N.op_crossed(p[p.method.isin(["mask", "erm"])], "mask", "erm", 20260928, hard_pos_a=1)
        pd.DataFrame(op_rows).to_csv(out / "ft_natural" / "operating_points.csv", index=False)
        sens = [r for r in op_rows if r["metric"] == "sens" and str(r["op"]).startswith(("OP2", "OP3", "OP4", "OP5"))]
        n_neg = sum(1 for r in sens if r["ci95_hi"] < 0)
        reps = _op_replicates(p[p.method.isin(["mask", "erm"])], 400 if smoke else 2000)
        op_names = [r["op"] for r in sens]
        if op_names:
            count = np.sum([(reps[o]["sens"] < 0).astype(int) for o in op_names], axis=0)
            p_ft2 = float(max((count < 3).mean(), 1.0 / len(count)))
        else:
            p_ft2 = 1.0
        ft2 = {"id": "FT2", "n_ops_ci_below_0": n_neg, "n_ops": len(sens),
               "estimate": n_neg, "p": p_ft2,
               "direction": "sensitivity CI < 0 at >= 3 of OP2-OP5",
               "ops": [{k: r[k] for k in ("op", "delta", "ci95_lo", "ci95_hi")} for r in sens]}
        hit = [r for r in op_rows if str(r["op"]).startswith("OP2") and r["metric"] == "sens_conflict"]
        if hit:
            r = hit[0]
            arr = reps[r["op"]]["sens_conflict"]
            ft3 = {"id": "FT3", "estimate": r["delta"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
                   "p": C.one_sided_p(arr, "less"),
                   "direction": "OP2 sensitivity among in-ROI-caliper malignancies, mask-erm < 0"}
    # FT4 from the fine-tune trap predictions
    arch = (choice or {}).get("arch", "resnet50")
    tag = "round6_smoke" if smoke else "round6"
    tdir = C.ROOT / "results" / "finetune" / "thyroid" / (arch.replace("/", "_") + f"_{tag}")
    ft4 = {"id": "FT4", "ran": False}
    tp = tdir / "predictions.csv.gz"
    if tp.exists():
        preds = pd.read_csv(tp)
        pb, pa = preds[preds.trap == "trapB"], preds[preds.trap == "trapA"]
        if len(pb) and len(pa) and {"mask", "erm"} <= set(preds.method):
            _point, arr = X.replicates(X._pair_terms(pb, "mask", "erm", "test_rev", "seed", +1.0)
                                       + X._pair_terms(pa, "mask", "erm", "test_rev", "seed", -1.0),
                                       2000 if smoke else 10000, 20260928)
            est = float(np.mean(list(_point.values())))
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ft4 = {"id": "FT4", "ran": True, "estimate": est, "ci95_lo": float(lo), "ci95_hi": float(hi),
                   "p": C.one_sided_p(arr, "greater"), "direction": "thyroid trap crossover > 0",
                   "n_clusters": len(_point)}
    family = [h for h in (ft1, ft2, ft3, ft4) if h.get("p") is not None]
    adj = {r["id"]: r["p_holm"] for r in C.holm_table(family, "p")}
    def verdict(h, positive_is_low: bool):
        if h.get("p") is None:
            return "not run"
        ph = adj.get(h["id"], 1.0)
        h["p_holm"] = ph
        if h["id"] == "FT2":
            return "SUPPORTED" if h.get("n_ops_ci_below_0", 0) >= 3 and ph < 0.05 else "NOT SUPPORTED"
        lo, hi, est = h.get("ci95_lo"), h.get("ci95_hi"), h.get("estimate")
        if positive_is_low:
            return "SUPPORTED" if est is not None and est < 0 and hi is not None and hi < 0 and ph < 0.05 else "NOT SUPPORTED"
        return "SUPPORTED" if est is not None and est > 0 and lo is not None and lo > 0 and ph < 0.05 else "NOT SUPPORTED"
    ft1["verdict"] = verdict(ft1, True)
    ft2["verdict"] = verdict(ft2, True)
    ft3["verdict"] = verdict(ft3, True)
    ft4["verdict"] = verdict(ft4, False)
    # also the descriptive paired contrasts on all/hard/easy
    extra = []
    for label, sel in (("hard", hard), ("easy", ~hard), ("all", hard | ~hard)):
        for a1, a0 in (("mask", "erm"), ("balanced", "mask"), ("mask_balanced", "mask")):
            if {a1, a0} <= set(te.method):
                qq = te[sel & te.method.isin([a1, a0])]
                if qq.seed.nunique() >= 1 and len(qq):
                    try:
                        pt, ar = X.replicates(X._pair_terms(qq, a1, a0, "clean", "seed"), 2000 if smoke else 10000, 20260928)
                        lo, hi = np.percentile(ar, [2.5, 97.5])
                        extra.append({"subset": label, "contrast": f"{a1}-{a0}", "estimate": float(np.mean(list(pt.values()))),
                                      "ci95_lo": float(lo), "ci95_hi": float(hi)})
                    except Exception as e:
                        extra.append({"subset": label, "contrast": f"{a1}-{a0}", "error": str(e)})
    rec = {"recipe": choice, "FT0": ft0, "FT1": ft1, "FT2": ft2, "FT3": ft3, "FT4": ft4, "contrasts": extra}
    (out / "ft_results.json").write_text(json.dumps(rec, indent=2, default=str))
    _write_ft_summary(out, rec)
    print((out / "ft_SUMMARY.md").read_text())


def _write_ft_summary(out: Path, rec: dict):
    lines = ["# Round 6 Part B — fine-tuned thyroid", ""]
    ch = rec.get("recipe") or {}
    lines.append(f"Recipe: `{ch.get('arch')}` lr={ch.get('lr')} epochs={ch.get('epochs')} validation AUROC {ch.get('val_auroc')}.")
    lines.append("")
    lines.append("| hypothesis | estimate | 95% CI | p | Holm p | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("FT0", "FT1", "FT2", "FT3", "FT4"):
        h = rec[key]
        est = h.get("estimate")
        ci = "" if h.get("ci95_lo") is None else f"[{h['ci95_lo']:+.3f}, {h['ci95_hi']:+.3f}]"
        lines.append(f"| {key} | {est if est is not None else ''} | {ci} | {h.get('p')} | {h.get('p_holm', '')} | {h.get('verdict')} |")
    lines.append("")
    lines.append("FT0 is descriptive (benchmark-level if the mean ERM test AUROC is at least 0.773) and is not in the Holm family.")
    lines.append("FT1–FT4 use the crossed bootstrap. Holm is applied to those four one-sided p-values.")
    lines.append("Hard pairs are malignant nodules with an in-ROI caliper versus benign nodules without (`hard_pos_a=1`).")
    (out / "ft_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--traps", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.select:
        select(a.smoke)
    elif a.natural:
        natural(a.smoke)
    elif a.traps:
        traps(a.smoke)
    elif a.analyse:
        analyse(a.smoke)
    else:
        ap.error("pass --select, --natural, --traps or --analyse")


if __name__ == "__main__":
    main()
