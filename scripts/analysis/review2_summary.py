"""Collect the second-review analyses (docs/PREREGISTRATION_REVIEW2.md) into result tables.

  python scripts/analysis/review2_summary.py   -> results/review2/{R0_repro,R1_matched,R1_balance,R2_transplant,
                                                   R4_finetune}.csv, summary.json, REVIEW2_RESULTS.md
Missing inputs are skipped (the script can be re-run as runs finish).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths
from wtss.stats import difference_of_deltas

R = paths.RESULTS
OUT = paths.ensure(R / "review2")
COHORTS = {"isic": ("spec_e13", "dino518_spec", "ISIC hair"), "thyroid": ("thyroid", "dino518_main", "Thyroid"),
           "ovary": ("ovary", "dino518_main", "Ovary"), "capsule": ("capsule", "dino518_main", "Capsule")}
REPRO_TOL = 0.03


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def crossover(d: Path):
    f = d / "predictions.csv.gz"
    if not f.exists():
        return None
    p = pd.read_csv(f)
    return difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)


def archived(c):
    """Archived primary P1 crossover (results/PRIMARY_CLAIMS.csv, pre-registered primary family)."""
    base, tag, _ = COHORTS[c]
    p = pd.read_csv(R / "PRIMARY_CLAIMS.csv")
    r = p[(p.family == "P1 location crossover") & (p.source == f"{base}/{tag}")].iloc[0]
    return {"seed_delta_mean": r.estimate, "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi}


def r0_r1():
    rows = []
    for c, (base, _, label) in COHORTS.items():
        a = archived(c)
        for tag in ("repro", "matched"):
            x = crossover(R / base / f"dino518_{tag}")
            if x is None:
                continue
            rows.append({"cohort": label, "run": tag, "crossover": x["seed_delta_mean"], "ci95_lo": x["ci95_lo"],
                         "ci95_hi": x["ci95_hi"], "p_boot_two_sided": x["p_boot_two_sided"],
                         "archived": a["seed_delta_mean"], "archived_lo": a["ci95_lo"], "archived_hi": a["ci95_hi"]})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    rep = d[d.run == "repro"].set_index("cohort").crossover
    d["repro_crossover"] = d.cohort.map(rep)
    d["abs_diff_vs_archived"] = (d.crossover - d.archived).abs()
    d["R0_pass"] = np.where(d.run == "repro", (np.sign(d.crossover) == np.sign(d.archived)) & (d.ci95_lo > 0)
                            & (d.abs_diff_vs_archived <= REPRO_TOL), None)
    m = d.run == "matched"
    d.loc[m, "ratio_matched_to_repro"] = d.loc[m, "crossover"] / d.loc[m, "repro_crossover"]
    # pre-registered feasibility: >= 30 matched images per label in each trap (pairs summed over source strata)
    base_of = {v[2]: v[0] for v in COHORTS.values()}
    for i in d.index[m]:
        st = pd.read_csv(R / base_of[d.at[i, "cohort"]] / "dino518_matched" / "match_strata.csv")
        st["y"] = st.stratum.str.extract(r"\((\d)")[0].astype(int)
        per = st.groupby("y").pairs.sum()
        d.at[i, "min_pairs_per_label"] = int(per.min())
        d.at[i, "feasible"] = bool(per.min() >= 30)
        bal = pd.read_csv(R / base_of[d.at[i, "cohort"]] / "dino518_matched" / "match_after.csv")
        d.at[i, "max_abs_smd_after_pooled"] = bal[bal.subset == "pooled"].smd.abs().max()
        bb = pd.read_csv(R / base_of[d.at[i, "cohort"]] / "dino518_matched" / "match_before.csv")
        d.at[i, "max_abs_smd_before_pooled"] = bb[bb.subset == "pooled"].smd.abs().max()
    if m.sum():  # Holm over the four cohorts; an infeasible cohort enters with p = 1 (conservative)
        pv = np.where(d.loc[m, "feasible"] == True, d.loc[m, "p_boot_two_sided"], 1.0)  # noqa: E712
        d.loc[m, "p_holm"] = holm(pv)
        d.loc[m, "M1_supported"] = np.where(d.loc[m, "feasible"] == True, (d.loc[m, "ci95_lo"] > 0) & (d.loc[m, "p_holm"] < 0.05), None)  # noqa: E712
    return d


def balance():
    """R1a: SMD Trap A vs Trap B artifact-bearing images (before / after matching) and each vs the artifact-free group."""
    from wtss.matching import covariates, smd
    spec = importlib.util.spec_from_file_location("rt", paths.REPO_ROOT / "scripts" / "run_thyroid_traps.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    from wtss.data.isic2019_spec import load_spec_cohort
    load = {"isic": load_spec_cohort, "thyroid": rt.cohort, "ovary": rt.ovary_cohort, "capsule": rt.capsule_cohort}
    rows = []
    for c, (base, _, label) in COHORTS.items():
        d = R / base / "dino518_matched"
        for k in ("before", "after"):
            f = d / f"match_{k}.csv"
            if f.exists():
                t = pd.read_csv(f).assign(cohort=label, comparison=f"trapA_vs_trapB_{k}_matching")
                rows.append(t)
        try:
            co = load[c]()
        except FileNotFoundError:
            continue
        co["image_id"] = co.image_id.astype(str)
        tab, num, cat = covariates(c, co)
        a0 = co[co.A0]
        for trap in ("trapA", "trapB"):
            u = pd.concat([co[co[f"{trap}_A1"]].assign(g=1), a0.assign(g=0)])
            s = smd(tab.loc[u.image_id], u.g.to_numpy(), num, cat)
            s.insert(0, "subset", "pooled")
            rows.append(s.assign(cohort=label, comparison=f"{trap}_A1_vs_artifact_free", n_trapA=int(u.g.sum()),
                                 n_trapB=int((1 - u.g).sum())))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def r2():
    rows = []
    for c, (_, _, label) in COHORTS.items():
        d = R / "review2" / "transplant" / f"{c}_dino518"
        if not (d / "location_interaction.csv").exists():
            continue
        li = pd.read_csv(d / "location_interaction.csv")
        bt = pd.read_csv(d / "bootstrap_vs_erm.csv")
        auc = pd.read_csv(d / "auc_per_seed.csv")
        cf = pd.read_csv(d / "counterfactual_summary.csv") if (d / "counterfactual_summary.csv").exists() else None
        des = json.loads((d / "design.json").read_text())
        mi = li[li.method == "mask"].iloc[0]
        row = {"cohort": label, "n_images": des["recipients_feasible"], "n_pos": des["n_pos"],
               "interaction": mi.seed_delta_mean, "ci95_lo": mi.ci95_lo, "ci95_hi": mi.ci95_hi,
               "p_boot_two_sided": mi.get("p_boot_two_sided", np.nan)}
        for ov, tag in ((1.0, "in"), (0.0, "out")):
            b = bt[(bt.method_a == "mask") & np.isclose(bt.overlap, ov)].iloc[0]
            row[f"mask_minus_erm_{tag}"], row[f"mask_minus_erm_{tag}_lo"], row[f"mask_minus_erm_{tag}_hi"] = \
                b.seed_delta_mean, b.ci95_lo, b.ci95_hi
            for m in ("erm", "mask", "balanced", "inpaint"):
                q = auc[(auc.method == m) & np.isclose(auc.overlap, ov)]
                for env in ("test_rev", "test_corr", "clean"):
                    row[f"auc_{m}_{env}_{tag}"] = q[q.env == env].auc.mean()
            if cf is not None:
                for m in ("erm", "mask"):
                    q = cf[(cf.method == m) & np.isclose(cf.overlap, ov)]
                    row[f"cf_absdp_{m}_{tag}"] = float(q.abs_delta_p.iloc[0]) if len(q) else np.nan
        bi = li[li.method == "balanced"]
        if len(bi):
            row["balanced_interaction"], row["balanced_lo"], row["balanced_hi"] = bi.seed_delta_mean.iloc[0], bi.ci95_lo.iloc[0], bi.ci95_hi.iloc[0]
        rows.append(row)
    d = pd.DataFrame(rows)
    if len(d):
        d["p_holm"] = holm(d.p_boot_two_sided.fillna(1.0))
        d["T1_supported"] = (d.ci95_lo > 0) & (d.p_holm < 0.05)
    return d


def r4():
    rows = []
    adhoc = json.loads((R / "adhoc_bootstraps.json").read_text())  # archived fine-tuned crossovers (5 folds of seed 42)
    for key, (c, arch) in {"finetune|thyroid_resnet50|crossover mask-erm (B-A)": ("thyroid", "resnet50"),
                           "finetune|thyroid_vit_s|crossover mask-erm (B-A)": ("thyroid", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
                           "finetune|capsule_resnet50|crossover mask-erm (B-A)": ("capsule", "resnet50"),
                           "finetune|ovary_resnet50|crossover mask-erm (B-A)": ("ovary", "resnet50")}.items():
        v = adhoc[key]
        rows.append({"cohort": c, "arch": arch, "clusters": 5, "run": "archived", "crossover": v[0], "ci95_lo": v[1],
                     "ci95_hi": v[2], "p_boot_two_sided": np.nan})
    for c in ("isic", "thyroid", "ovary", "capsule"):
        for arch in ("resnet50", "resnet50_power", "vit_small_patch16_224.augreg_in21k_ft_in1k"):
            f = R / "finetune" / c / arch / "predictions.csv.gz"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            if not {"erm", "mask"} <= set(p.method) or p.trap.nunique() < 2:
                continue
            x = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
            auc = p.groupby(["trap", "method", "env", "seed"]).apply(lambda q: _auc(q), include_groups=False).groupby(
                ["trap", "method", "env"]).mean()
            rows.append({"cohort": c, "arch": arch, "clusters": int(p.seed.nunique()), "run": "review2",
                         "crossover": x["seed_delta_mean"], "ci95_lo": x["ci95_lo"],
                         "ci95_hi": x["ci95_hi"], "p_boot_two_sided": x["p_boot_two_sided"],
                         "erm_clean_trapA": auc.get(("trapA", "erm", "clean"), np.nan),
                         "erm_clean_trapB": auc.get(("trapB", "erm", "clean"), np.nan)})
    return pd.DataFrame(rows)


def _auc(q):
    from wtss.stats import safe_auc
    return safe_auc(q.y, q.prob)


def natural_repro():
    rows = []
    for c in ("thyroid", "isic_BCN", "isic_HAM", "isic_MSK", "capsule"):
        a, b = R / "natural" / f"{c}_dino518" / "natural_boot.json", R / "natural" / f"{c}_dino518_repro" / "natural_boot.json"
        if not (a.exists() and b.exists()):
            continue
        ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
        for k in ("hard | mask-erm", "all | mask-erm", "hard | mte_balanced-mask"):
            if k in ja and k in jb:
                rows.append({"cohort": c, "contrast": k, "archived": ja[k][0], "archived_lo": ja[k][1], "archived_hi": ja[k][2],
                             "repro": jb[k][0], "repro_lo": jb[k][1], "repro_hi": jb[k][2]})
    return pd.DataFrame(rows)


def main():
    res = {}
    for name, fn in (("R0_R1_crossovers", r0_r1), ("R1_balance", balance), ("R2_transplant", r2),
                     ("R4_finetune", r4), ("R0_natural", natural_repro)):
        d = fn()
        if len(d):
            d.to_csv(OUT / f"{name}.csv", index=False)
            res[name] = d.round(4).to_dict("records") if name != "R1_balance" else "R1_balance.csv"
            print(f"== {name}\n{d.round(3).to_string()}\n" if name != "R1_balance" else f"== {name}: {len(d)} rows")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
