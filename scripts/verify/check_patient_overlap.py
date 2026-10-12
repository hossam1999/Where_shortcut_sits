"""Check on the real NIH ChestX-ray14 data that no patient appears in two splits or folds of any chest X-ray experiment.

Run on the machine that holds the prepared chest X-ray data:
  PYTHONPATH=src python scripts/verify/check_patient_overlap.py
  -> results/verification/patient_overlap.csv (one row per check; n_shared_patients must be 0)

Patient identifiers are NIH's "Patient ID" column of Data_Entry_2017_v2020.csv; the script also checks that each image
name starts with its patient's 8-digit identifier.
"""
from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths
from wtss.data.cxr import PREP, nih_table
from wtss.data.cxr_drain import build_drain_envs, load_drain_cohort
from wtss.data.isic2019_spec import SEEDS, build_spec_envs

ROOT = Path(__file__).resolve().parents[2]


def _shared(sets):
    return int(sum(len(a & b) for a, b in combinations(sets, 2)))


def main():
    rows = []
    nih = nih_table()
    prefix_ok = (nih.image_id.str[:8].astype(int) == nih.patient.astype(int))
    rows.append({"experiment": "NIH metadata", "check": "image name starts with Patient ID", "n_images": len(nih),
                 "n_patients": nih.patient.nunique(), "n_shared_patients": int((~prefix_ok).sum())})
    pid = nih.set_index("image_id").patient.astype(str)

    # moving-tube experiment: one fixed train / val / test split
    c = pd.read_csv(PREP / "nih_ptx_cohort.csv")
    p = pid.loc[c.image_id]
    sets = [set(p[(c.split == s).to_numpy()]) for s in ("train", "val", "test")]
    rows.append({"experiment": "moving tube (nih_ptx)", "check": "train/val/test", "n_images": len(c),
                 "n_patients": p.nunique(), "n_shared_patients": _shared(sets)})

    # chest drains: 5 patient-grouped folds; each fold k is tested, fold k+1 validates, the rest trains
    d = load_drain_cohort()
    envs, _ = build_drain_envs(d)
    p = pid.loc[d.image_id]
    rows.append({"experiment": "chest drains", "check": "the 5 test folds", "n_images": len(d),
                 "n_patients": p.nunique(), "n_shared_patients": _shared([set(p[(d.fold == k).to_numpy()]) for k in range(5)])})
    for k in range(5):
        tr = set(pid.loc[envs[("drain", k, "train_all")].image_id])
        va = set(pid.loc[envs[("drain", k, "val_groups")].image_id])
        te = set(pid.loc[envs[("drain", k, "test_rev")].image_id]) | set(pid.loc[envs[("drain", k, "clean_test")].image_id])
        rows.append({"experiment": "chest drains", "check": f"fold {k}: train/val/test", "n_images": np.nan,
                     "n_patients": len(tr | va | te), "n_shared_patients": _shared([tr, va, te])})

    # device traps: 5 seeds x 5 patient-grouped folds per finding and trap
    spec = importlib.util.spec_from_file_location("rct", ROOT / "scripts" / "run_cxr_traps.py")
    rct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rct)
    for dis in rct.DISEASES:
        cd = rct.cohort(dis)
        traps = tuple(rct.TRAPS)
        envs = build_spec_envs(cd, traps=traps, group_col="patient")
        pp = cd.set_index("image_id").patient.astype(str)
        for trap in traps:
            worst = 0
            for seed in SEEDS:
                tests = []
                for k in range(5):
                    tr = set(pp.loc[envs[(trap, seed, k, "train_all")].image_id])
                    va = set(pp.loc[envs[(trap, seed, k, "val_groups")].image_id])
                    te = set(pp.loc[envs[(trap, seed, k, "test_rev")].image_id]) | set(
                        pp.loc[envs[(trap, seed, k, "clean")].image_id])
                    worst = max(worst, _shared([tr, va, te]))
                    tests.append(te)
                worst = max(worst, _shared(tests))
            rows.append({"experiment": f"device traps ({dis})", "check": f"{trap}: 5 seeds x 5 folds", "n_images": len(cd),
                         "n_patients": cd.patient.nunique(), "n_shared_patients": worst})

    out = pd.DataFrame(rows)
    dest = paths.ensure(paths.RESULTS / "verification") / "patient_overlap.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\n{'PASS' if (out.n_shared_patients == 0).all() else 'FAIL'}: {dest}")


if __name__ == "__main__":
    main()
