"""Round 8, Phase 1 step 2: the scoreboard of numbers to beat, from existing result files only.

Every row is recomputed from saved per-image predictions with the crossed seed x image bootstrap (one set of shared
replicates per source run, scripts/round8/common.joint_replicates), so every arm can be compared with masking on the
same replicates; rows that exist only as summary files (operating points, round 7 matched thresholds) are copied.
No model is trained and no new test set is scored.

    python scripts/round8/scoreboard.py [--smoke] [--n-boot 10000] [--jobs 12]

Output: results/round8/scoreboard_existing.csv (+ scoreboard_parts/ per source, resumable; repro_check.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r8(name):
    """Load a round-8 module by path under a unique name (earlier rounds also have a module called `common`)."""
    import importlib.util
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


C = _r8("common")
R4 = C.load_round4_common()

ENCODER = {"dino518": "DINOv2-B/14@518", "dinol518": "DINOv2-L/14@518", "dinos518": "DINOv2-S/14@518",
           "medsiglip448": "MedSigLIP@448", "convnext384": "ConvNeXt@384", "dermlip224": "DermLIP@224",
           "dino224": "DINOv2-B/14@224", "raddino518": "RAD-DINO@518", "convnext_tiny_ft": "ConvNeXt-T fine-tuned"}
COHORT = {"thyroid": "thyroid", "capsule": "capsule", "ovary": "ovary", "spec_e13": "isic_hair"}
TRAP_SKIP = ("matched", "repro", "emb_groups", "sens_", "ablation", "e12_contrast")  # robustness variants, not remedies
ENVS = ("test_rev", "test_corr", "clean")


def arm_name(run: str, method: str) -> str:
    """U-MtE and template MtE share the method name 'mte' in the stored files; make the arm explicit."""
    template_run = run.endswith("_main") or run in ("dino518_spec", "dermlip224_spec")
    if method.startswith("mte"):
        return ("mte_tpl" if template_run else "umte") + method[3:]
    return method


def sources(smoke: bool) -> list[dict]:
    S = []
    for d in sorted((C.RERUN / "synthetic").glob("*/*")):
        S.append(dict(setting="S1", cohort=f"synthetic_{d.parent.name}", run=d.name, kind="sweep",
                      pred=d / "predictions.csv.gz", encoder=d.name.split("_")[0]))
    for coh in ("thyroid", "capsule", "ovary", "spec_e13"):
        for d in sorted((C.RERUN / coh).iterdir()):
            if any(t in d.name for t in TRAP_SKIP) or not (d / "predictions.csv.gz").exists():
                continue
            S.append(dict(setting="S2", cohort=COHORT[coh], run=d.name, kind="trap", pred=d / "predictions.csv.gz",
                          encoder=d.name.split("_")[0]))
    for d in sorted((C.RERUN / "natural").iterdir()):
        coh = d.name.replace("_dino518_repro", "")
        S.append(dict(setting="S3", cohort=coh, run=d.name, kind="natural", pred=d / "predictions.csv.gz",
                      encoder="dino518", boot=d / "natural_boot.json"))
    S.append(dict(setting="S4", cohort="isic2019_to_2020", run="round5_calibrated", kind="isic2020",
                  pred=C.ROOT / "results/round5/predictions.csv.gz", encoder="dino518"))
    S.append(dict(setting="S4", cohort="isic2020_traps", run="external_isic2020/traps_dino518", kind="trap",
                  pred=C.RERUN / "external_isic2020/traps_dino518/predictions.csv.gz", encoder="dino518"))
    S.append(dict(setting="S5", cohort="thyroid", run="round6_ft_natural", kind="ft_natural",
                  pred=C.ROOT / "results/round6/ft_natural/predictions.csv.gz", encoder="convnext_tiny_ft"))
    S.append(dict(setting="S5", cohort="thyroid", run="round6_ft_traps", kind="trap",
                  pred=C.ROOT / "results/finetune/thyroid/convnext_tiny.fb_in22k_ft_in1k_round6/predictions.csv.gz",
                  encoder="convnext_tiny_ft"))
    if smoke:
        keep = {"dino518_caliper_corr_main", "dino518_universal", "thyroid_dino518_repro", "round6_ft_natural"}
        S = [s for s in S if s["run"] in keep and s["cohort"] in ("synthetic_thyroid", "thyroid")]
    return S


def natural_hard(p: pd.DataFrame, rule: str) -> pd.Series:
    """Shortcut-conflicting pairs, exactly as the scripts that produced the stored numbers.
    rule 'prevalence' (scripts/run_natural.py): the pairing where the artifact is rarer in the class it marks;
    'hair' (scripts/round5/isic2020_robust.py): melanoma without / benign with in-lesion hair;
    'caliper' (scripts/round6/ft_thyroid.py): malignant with / benign without an in-ROI caliper."""
    y, a = p.y, p.artifact_present
    if rule == "prevalence":
        one = p.drop_duplicates("image_id")
        pa1, pa0 = one[one.y == 1].artifact_present.mean(), one[one.y == 0].artifact_present.mean()
        rule = "caliper" if pa1 < pa0 else "hair"
    if rule == "caliper":
        return ((y == 1) & (a == 1)) | ((y == 0) & (a == 0))
    return ((y == 1) & (a == 0)) | ((y == 0) & (a == 1))


def _row(src, cell: dict, metric, arm, est, lo, hi, n_clusters, extra=None):
    return {"setting": src["setting"], "cohort": src["cohort"], "encoder": ENCODER.get(src["encoder"], src["encoder"]),
            "run": src["run"], **cell, "metric": metric, "arm": arm, "estimate": est, "ci95_lo": lo, "ci95_hi": hi,
            "n_clusters": n_clusters, "source_file": str(Path(src["pred"]).relative_to(C.ROOT)),
            "note": "recomputed from saved predictions, crossed bootstrap" if extra is None else extra}


def score_source(job) -> pd.DataFrame:
    src, n_boot, out = job
    part = out / "scoreboard_parts" / f"{src['setting']}__{src['cohort']}__{src['run'].replace('/', '_')}.csv"
    if part.exists():
        return pd.read_csv(part)
    p = pd.read_csv(src["pred"])
    p["image_id"] = p.image_id.astype(str)
    p["arm"] = [arm_name(src["run"], m) for m in p.method]
    kind = src["kind"]
    if kind == "sweep":
        p = p[p.overlap.isin([0.0, 1.0])]
        p["cell"] = "r=" + p.overlap.map(lambda v: f"{v:g}")
        p = p[p.env.isin(ENVS)]
    elif kind == "trap":
        p["cell"] = p.trap
        p = p[p.env.isin(ENVS)]
    else:  # natural test sets: all / hard / easy pairs of the clean (natural) test set
        p = p[p.env == "clean"]
        if kind == "isic2020":
            tiers = pd.read_csv(C.ROOT / "results/round5/dedup_tiers.csv", usecols=["image_id", "drop_calibrated"])
            keep = set(tiers.loc[~tiers.drop_calibrated.astype(bool), "image_id"].astype(str))
            p = p[p.image_id.isin(keep)]
        rule = {"natural": "prevalence", "isic2020": "hair", "ft_natural": "caliper"}[kind]
        hard = natural_hard(p, rule)
        p = pd.concat([p.assign(cell="all"), p[hard].assign(cell="hard"), p[~hard].assign(cell="easy")])
    seed = C.stable_seed("round8_scoreboard", src["setting"], src["cohort"], src["run"])
    point, reps = C.joint_replicates(C.terms_from_predictions(p, ["cell", "arm", "env"]), n_boot, seed)
    rows = []
    ncl = p.groupby(["cell", "arm", "env"]).seed.nunique().to_dict()
    for (cell, arm, env), v in point.items():
        lo, hi = C.ci(reps[(cell, arm, env)])
        c = {"cell": cell, "env": env}
        rows.append(_row(src, c, "auroc", arm, v, lo, hi, ncl[(cell, arm, env)]))
        for ref in ("mask", "erm"):
            if arm != ref and (cell, ref, env) in point:
                d = reps[(cell, arm, env)] - reps[(cell, ref, env)]
                lo, hi = C.ci(d)
                rows.append(_row(src, c, f"delta_vs_{ref}", arm, v - point[(cell, ref, env)], lo, hi,
                                 ncl[(cell, arm, env)]))
    if kind in ("sweep", "trap"):  # min(rev, corr): the robustness summary, and its contrast with masking
        for cell, arm in {(k[0], k[1]) for k in point}:
            kr, kc = (cell, arm, "test_rev"), (cell, arm, "test_corr")
            if kr not in point or kc not in point:
                continue
            m = np.fmin(reps[kr], reps[kc])
            est = min(point[kr], point[kc])
            lo, hi = C.ci(m)
            c = {"cell": cell, "env": "min_rev_corr"}
            rows.append(_row(src, c, "auroc", arm, est, lo, hi, ncl[kr]))
            mr, mc = (cell, "mask", "test_rev"), (cell, "mask", "test_corr")
            if arm != "mask" and mr in point and mc in point:
                d = m - np.fmin(reps[mr], reps[mc])
                lo, hi = C.ci(d)
                rows.append(_row(src, c, "delta_vs_mask", arm, est - min(point[mr], point[mc]), lo, hi, ncl[kr]))
    df = pd.DataFrame(rows)
    part.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(part, index=False)
    C.log(scored=src["run"], cohort=src["cohort"], rows=len(df))
    return df


def copied_rows() -> pd.DataFrame:
    """Operating points (mask vs ERM only; no other arm has them) and round 7 matched thresholds."""
    rows = []
    f = C.RERUN / "final_op" / "operating_points_crossed_all.csv"
    op = pd.read_csv(f)
    for r in op.itertuples():
        setting = "S4" if r.cohort.startswith("isic2020") else "S3"
        rows.append({"setting": setting, "cohort": r.cohort, "encoder": ENCODER["dino518"], "run": "final_op",
                     "cell": r.op, "env": "natural", "metric": f"{r.metric}_delta_vs_erm", "arm": "mask",
                     "estimate": r.delta, "ci95_lo": r.crossed_lo, "ci95_hi": r.crossed_hi, "n_clusters": 5,
                     "source_file": str(f.relative_to(C.ROOT)), "note": "copied (operating point; mask and ERM only)"})
    f = C.ROOT / "results" / "round7" / "matched_thresholds.csv"
    if f.exists():
        mt = pd.read_csv(f)
        for r in mt.to_dict("records"):
            if r.get("model") != "finetuned_round6":
                continue
            rows.append({"setting": "S5", "cohort": r["cohort"], "encoder": ENCODER["convnext_tiny_ft"],
                         "run": "round7_matched", "cell": r["match"], "env": "natural",
                         "metric": f"{r['metric']}_delta_vs_erm", "arm": "mask", "estimate": r.get("delta"),
                         "ci95_lo": r.get("ci95_lo"), "ci95_hi": r.get("ci95_hi"), "n_clusters": np.nan,
                         "source_file": str(f.relative_to(C.ROOT)),
                         "note": "copied (threshold matched on the test set)"})
    return pd.DataFrame(rows)


def mark_best(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: the arm with the highest AUROC (or min(rev, corr)), and whether it beats masking (CI > 0)."""
    key = ["setting", "cohort", "encoder", "cell", "env"]
    a = df[df.metric == "auroc"]
    best = a.loc[a.groupby(key).estimate.idxmax(), key + ["arm", "run"]].rename(columns={"arm": "best_arm",
                                                                                      "run": "best_arm_run"})
    dm = df[df.metric == "delta_vs_mask"].merge(best, on=key)
    dm = dm[(dm.arm == dm.best_arm) & (dm.run == dm.best_arm_run)]
    best = best.merge(dm[key + ["estimate", "ci95_lo"]].rename(columns={"estimate": "best_minus_mask",
                                                                         "ci95_lo": "best_minus_mask_lo"}),
                      on=key, how="left")
    return df.merge(best, on=key, how="left")


def repro_check(df: pd.DataFrame, srcs) -> pd.DataFrame:
    """Point estimates must match the stored files (bootstrap_vs_erm.csv seed means; natural_boot.json)."""
    rows = []
    for s in srcs:
        d = df[(df.run == s["run"]) & (df.cohort == s["cohort"])]
        f = Path(s["pred"]).parent / "bootstrap_vs_erm.csv"
        if s["kind"] == "trap" and f.exists():
            b = pd.read_csv(f)
            if "source" in b:  # ISIC runs also store per-source strata; the scoreboard pools the sources
                b = b[b.source == "all"]
            for r in b.itertuples():
                arm = arm_name(s["run"], r.method_a)
                m = d[(d.cell == r.trap) & (d.env == r.env) & (d.arm == arm) & (d.metric == "delta_vs_erm")]
                if len(m):
                    rows.append({"run": s["run"], "cohort": s["cohort"], "what": f"{r.trap} {r.env} {arm}-erm",
                                 "stored": r.seed_delta_mean, "recomputed": float(m.estimate.iloc[0])})
        if s["kind"] == "natural" and Path(s["boot"]).exists():
            b = json.loads(Path(s["boot"]).read_text())
            for k, v in b.items():
                if "|" not in k or "-" not in k.split("|")[1]:
                    continue
                sub, con = [x.strip() for x in k.split("|")]
                arm, ref = con.split("-")
                arm = arm_name(s["run"], arm)
                m = d[(d.cell == sub) & (d.arm == arm) & (d.metric == f"delta_vs_{ref}")]
                if len(m):
                    rows.append({"run": s["run"], "cohort": s["cohort"], "what": f"{sub} {arm}-{ref}",
                                 "stored": v[0], "recomputed": float(m.estimate.iloc[0])})
    r = pd.DataFrame(rows)
    if len(r):
        r["abs_diff"] = (r.stored - r.recomputed).abs()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    out = C.out_dir(a.smoke)
    n_boot = 200 if a.smoke else a.n_boot
    srcs = sources(a.smoke)
    jobs = [(s, n_boot, out) for s in srcs]
    parts = R4.parallel_map(score_source, jobs) if a.jobs > 1 else [score_source(j) for j in jobs]
    df = pd.concat(parts + ([] if a.smoke else [copied_rows()]), ignore_index=True)
    df = mark_best(df)
    df.to_csv(out / "scoreboard_existing.csv", index=False, float_format="%.4f")
    rc = repro_check(df, srcs)
    rc.to_csv(out / "scoreboard_repro_check.csv", index=False, float_format="%.4f")
    C.log(rows=len(df), repro_max_abs_diff=float(rc.abs_diff.max()) if len(rc) else None,
          repro_n=len(rc), out=str(out / "scoreboard_existing.csv"))


if __name__ == "__main__":
    main()
