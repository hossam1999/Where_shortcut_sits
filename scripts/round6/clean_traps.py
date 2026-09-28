"""A2 — Stage 3 traps on labels no informative source contradicts.

  python scripts/round6/clean_traps.py --cohort thyroid
  python scripts/round6/clean_traps.py --summary
  python scripts/round6/clean_traps.py --cohort thyroid --rule consensus   # exploratory, not registered
The consensus rule (exploratory, decided after the registered rule failed the thyroid count gate) removes an image
only when the two model labellers place it in the same cell and that cell contradicts ours. Its output goes to
results/round6/exploratory/consensus_traps/<cohort>/ and never into the registered tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from wtss.data.isic2019_spec import SEEDS, build_spec_envs, spec_counts  # noqa: E402
from wtss.experiments.spec_traps import run_spec  # noqa: E402
from wtss.stats import difference_of_deltas  # noqa: E402


CONSENSUS_COLS = ("cell_bus", "cell_mg")


def contradicted(per: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "registered":
        return per.contradicted.astype(bool)
    a, b = (per[k].astype(str) for k in CONSENSUS_COLS)
    return pd.Series([x == y and C.contradicts(o, x) for o, x, y in zip(per.cell.astype(str), a, b)], index=per.index)


def dest_of(name: str, smoke: bool, rule: str) -> Path:
    if rule == "registered":
        return C.out_root(smoke) / "clean_traps" / name
    return C.out_root(smoke) / "exploratory" / "consensus_traps" / name


def cleaned_cohort(name: str, smoke: bool, rule: str = "registered") -> tuple[pd.DataFrame, pd.DataFrame]:
    info = C.cohort_frame(name)
    c = info["c"]
    per_p = C.out_root(smoke) / "agreement" / name / "per_image.csv"
    if not per_p.exists():
        raise SystemExit(f"missing {per_p}; run label_agreement first")
    per = pd.read_csv(per_p)
    if rule == "consensus" and not set(CONSENSUS_COLS) <= set(per.columns):
        raise SystemExit(f"the consensus rule needs {CONSENSUS_COLS} in {per_p}")
    per["image_id"] = per.image_id.astype(str)
    c["image_id"] = c.image_id.astype(str)
    keep_ids = set(per.loc[~contradicted(per, rule), "image_id"])
    out = c[c.image_id.isin(keep_ids)].reset_index(drop=True)
    if smoke:
        out = C.R4.smoke_subset(out, ("trapA", "trapB"), n=40)
    return info, out


def _shares(per: pd.DataFrame, rule: str = "registered") -> pd.DataFrame:
    rows = []
    con = contradicted(per, rule)
    for cell, g in per.groupby("cell"):
        n = len(g)
        rows.append({"cell": cell, "n": n,
                     "verified": float(g.verified.mean()) if n else 0,
                     "contradicted": float(con[g.index].mean()) if n else 0,
                     "unverified": float(g.unverified.mean()) if n else 0})
    return pd.DataFrame(rows)


def _gate(counts: pd.DataFrame) -> dict:
    """Stage 7: every matched A x Y cell >= 25 and >= 40 reversed-test positives (seed 42), per trap."""
    ok = {}
    for r in counts.itertuples(index=False):
        cells = [getattr(r, f"A{a}_Y{y}") for a in (0, 1) for y in (0, 1)]
        ok[r.trap] = bool(min(cells) >= 25 and (r.rev_mel_seed42 or 0) >= 40)
    return ok


def run_cohort(name: str, smoke: bool, rule: str = "registered"):
    dest = dest_of(name, smoke, rule)
    dest.mkdir(parents=True, exist_ok=True)
    info, c = cleaned_cohort(name, smoke, rule)
    per = pd.read_csv(C.out_root(smoke) / "agreement" / name / "per_image.csv")
    shares = _shares(per, rule)
    shares.to_csv(dest / "shares.csv", index=False)
    seeds = (42,) if smoke else SEEDS
    envs = build_spec_envs(c, seeds=seeds, group_col=info["group_col"])
    counts = spec_counts(c, envs if not smoke else build_spec_envs(c, seeds=(42,), group_col=info["group_col"]))
    # spec_counts looks up seed 42; smoke envs already use seed 42
    if smoke:
        counts = spec_counts(c, envs)
    counts.to_csv(dest / "counts.csv", index=False)
    gate = _gate(counts)
    (dest / "gate.json").write_text(json.dumps(gate, indent=2))
    orig = C.original_crossover(name)
    if smoke or all(gate.values()):
        traps = ("trapA", "trapB")
        if not traps:
            C.log(cohort=name, gate="failed", detail=gate)
            (dest / "crossover.json").write_text(json.dumps({"gate": gate, "original": orig, "ran": False}, indent=2))
            return
        fdir = C.FEAT / ("_smoke" if smoke else "") / ("clean_traps" if rule == "registered" else "consensus_traps") \
            / name / C.R4.BDIR
        fdir.mkdir(parents=True, exist_ok=True)
        pool = C.R4.pool_of(envs, traps)
        sources = {}
        olds = list(info["old"]) + [C.R4.FEAT / "masking_variants" / name / C.R4.BDIR,
                                    C.R4.FEAT / "dose_response" / name / C.R4.BDIR]
        for v in ("erm", "mask"):
            sources[v] = [(p / f"{v}.npz", lambda i: i) for p in olds]
        from wtss.experiments.real_traps import make_renderers
        rend = make_renderers(info["cache"], 518, info["donors"])

        def extract():
            C.R4.assemble_views(pool, ["erm", "mask"], rend, fdir, sources, torch.device("cuda"), batch_size=128, workers=6)

        def fit():
            folds = [0] if smoke else range(5)
            run_spec(envs, info["cache"], C.R4.BACKBONE, dest, fdir, info["donors"], arms=("erm", "mask"),
                     traps=traps, device=torch.device("cuda"), n_jobs=C.R4.n_cpus(), folds=folds)
            _crossover(dest, orig, gate)

        if C.R4.views_ready(pool, ["erm", "mask"], fdir):
            fit()
        else:
            # extraction in a child; the parent fits. The child re-enters main via argv, so stop it after extract.
            if os_extract_only():
                extract()
                return
            C.R4.gpu_then_cpu(True, extract, fit)
    else:
        (dest / "crossover.json").write_text(json.dumps({"gate": gate, "original": orig, "ran": False}, indent=2))
        C.log(cohort=name, gate="failed")


def os_extract_only() -> bool:
    return __import__("os").environ.get("WTSS_EXTRACT_ONLY") == "1"


def _crossover(dest: Path, orig: dict, gate: dict):
    preds = pd.read_csv(dest / "predictions.csv.gz")
    pb, pa = preds[preds.trap == "trapB"], preds[preds.trap == "trapA"]
    if len(pb) == 0 or len(pa) == 0:
        rec = {"ran": False, "gate": gate, "original": orig}
    else:
        # Registered estimator. The replicate array is recomputed with the same call so the one-sided p is defined.
        r = difference_of_deltas(pb, pa, "mask", "erm", "test_rev", 10000, 20260928)
        from wtss import stats_crossed as X
        _point, arr = X.replicates(X._pair_terms(pb, "mask", "erm", "test_rev", "seed", +1.0)
                                   + X._pair_terms(pa, "mask", "erm", "test_rev", "seed", -1.0), 10000, 20260928)
        rec = {"ran": True, "gate": gate, "original": orig, "estimate": r["seed_delta_mean"],
               "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "p_greater": C.one_sided_p(arr, "greater"),
               "p_two_sided": r["p_boot_two_sided"], "n_boot": int(r["n_boot_valid"])}
    (dest / "crossover.json").write_text(json.dumps(rec, indent=2))
    C.log(cohort=dest.name, crossover=rec.get("estimate"))


def summary(smoke: bool):
    rows = []
    root = C.out_root(smoke) / "clean_traps"
    for name in C.COHORTS:
        p = root / name / "crossover.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if not d.get("ran"):
            rows.append({"cohort": name, "ran": False, "p": 1.0, "estimate": np.nan, "verdict": "gate not met"})
            continue
        rows.append({"cohort": name, "ran": True, "estimate": d["estimate"], "ci95_lo": d["ci95_lo"], "ci95_hi": d["ci95_hi"],
                     "p": d["p_greater"]})
    tested = [r for r in rows if r.get("ran")]
    adj = C.holm_table(tested, "p")
    by = {r["cohort"]: r for r in adj}
    out_rows = []
    for r in rows:
        if r["cohort"] in by:
            a = by[r["cohort"]]
            a["verdict"] = "SUPPORTED" if a["estimate"] > 0 and a["ci95_lo"] > 0 and a["p_holm"] < 0.05 else "NOT SUPPORTED"
            out_rows.append(a)
        else:
            out_rows.append(r)
    df = pd.DataFrame(out_rows)
    dest = C.out_root(smoke) / "clean_traps"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "SUMMARY.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=C.COHORTS)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rule", choices=("registered", "consensus"), default="registered")
    a = ap.parse_args()
    if os_extract_only() and a.cohort:
        run_cohort(a.cohort, a.smoke, a.rule)
        return
    if a.summary:
        summary(a.smoke)
    elif a.cohort:
        run_cohort(a.cohort, a.smoke, a.rule)
    else:
        ap.error("pass --cohort or --summary")


if __name__ == "__main__":
    main()
