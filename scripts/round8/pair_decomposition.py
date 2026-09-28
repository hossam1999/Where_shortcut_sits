"""Round 8, Phase 1 step 4: all-pairs AUROC as a mixture of pair types (existing predictions only; no new scoring).

For a positive i and a negative j, AUROC over all pairs = sum_k pi_k AUROC_k, where k is the pair type by artifact status
(easy: the pair the training association orders correctly; hard: the conflicting pair; same_a1 / same_a0: both carry /
both lack the artifact) and pi_k is the share of positive-negative pairs of that type. A remedy that removes the shortcut
gains on hard pairs, loses on easy pairs, and changes the same-artifact pairs only through the disease signal, so

    Delta_all = pi_easy * Delta_easy + pi_hard * Delta_hard + pi_same * Delta_same            (exact, per seed)

    python scripts/round8/pair_decomposition.py [--smoke]
Output: results/round8/pair_decomposition.csv (per cohort x arm: pair shares, per-type deltas vs masking, the sum).
"""
from __future__ import annotations

import argparse
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
SB = _r8("scoreboard")
from wtss.stats import binary_auc  # noqa: E402


def pair_auc(yp, sp, yn, sn):
    if len(sp) == 0 or len(sn) == 0:
        return np.nan, 0
    return binary_auc(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]), len(sp) * len(sn)


def decompose(q: pd.DataFrame, marker_of_positive: int):
    """marker_of_positive: artifact status that the training association links to y=1 (1 for hair, 0 for calipers)."""
    pos, neg = q[q.y == 1], q[q.y == 0]
    m = marker_of_positive
    types = {"easy": (m, 1 - m), "hard": (1 - m, m), "same_a1": (1, 1), "same_a0": (0, 0)}
    out = {}
    for k, (ap, an) in types.items():
        a, n = pair_auc(1, pos[pos.artifact_present == ap].prob.to_numpy(), 0, neg[neg.artifact_present == an].prob.to_numpy())
        out[k] = (a, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = C.out_dir(a.smoke)
    srcs = [s for s in SB.sources(a.smoke) if s["kind"] in ("natural", "isic2020", "ft_natural")]
    rows = []
    for s in srcs:
        p = pd.read_csv(s["pred"])
        p = p[p.env == "clean"].copy()
        p["image_id"] = p.image_id.astype(str)
        p["arm"] = [SB.arm_name(s["run"], m) for m in p.method]
        if s["kind"] == "isic2020":
            tiers = pd.read_csv(C.ROOT / "results/round5/dedup_tiers.csv", usecols=["image_id", "drop_calibrated"])
            p = p[p.image_id.isin(set(tiers.loc[~tiers.drop_calibrated.astype(bool), "image_id"].astype(str)))]
        rule = {"natural": "prevalence", "isic2020": "hair", "ft_natural": "caliper"}[s["kind"]]
        hard = SB.natural_hard(p, rule)
        # marker of the positive class: in a hard pair the positive lacks it
        mpos = int(p.loc[hard & (p.y == 1), "artifact_present"].iloc[0] == 0)
        per = []
        for (arm, seed), q in p.groupby(["arm", "seed"]):
            d = decompose(q, mpos)
            tot = sum(n for _, n in d.values())
            all_auc = binary_auc(q.y.to_numpy(), q.prob.to_numpy())
            per.append({"arm": arm, "seed": seed, "all": all_auc, **{f"auc_{k}": v[0] for k, v in d.items()},
                        **{f"pi_{k}": v[1] / tot for k, v in d.items()}})
        per = pd.DataFrame(per)
        mask = per[per.arm == "mask"].set_index("seed")
        for arm, g in per.groupby("arm"):
            if arm == "mask":
                continue
            g = g.set_index("seed")
            r = {"setting": s["setting"], "cohort": s["cohort"], "arm": arm, "marker_of_positive": mpos}
            for k in ("easy", "hard", "same_a1", "same_a0"):
                r[f"pi_{k}"] = float(g[f"pi_{k}"].mean())
                r[f"d_{k}"] = float((g[f"auc_{k}"] - mask[f"auc_{k}"]).mean())
            r["d_all_observed"] = float((g["all"] - mask["all"]).mean())
            r["d_all_from_parts"] = float(np.nansum([(g[f"pi_{k}"] * (g[f"auc_{k}"] - mask[f"auc_{k}"])).mean()
                                                     for k in ("easy", "hard", "same_a1", "same_a0")]))
            r["contrib_easy"] = float((g.pi_easy * (g.auc_easy - mask.auc_easy)).mean())
            r["contrib_hard"] = float((g.pi_hard * (g.auc_hard - mask.auc_hard)).mean())
            r["contrib_same"] = r["d_all_from_parts"] - r["contrib_easy"] - r["contrib_hard"]
            rows.append(r)
        C.log(decomposed=s["cohort"])
    df = pd.DataFrame(rows)
    df.to_csv(out / "pair_decomposition.csv", index=False, float_format="%.4f")
    C.log(rows=len(df), max_abs_identity_error=float((df.d_all_observed - df.d_all_from_parts).abs().max()))


if __name__ == "__main__":
    main()
