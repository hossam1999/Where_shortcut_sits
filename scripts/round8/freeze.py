"""Round 8: freeze the development choices before any confirmation run (docs/PREREGISTRATION_ROUND8.md, section 5).

  python scripts/round8/freeze.py [--smoke]      -> results/round8/frozen_choice.json (commit and push it before confirm)

Development used validation data only. What is frozen: the candidate list, grids, selection rule and guard margin,
seeds, and the settings of the held-out cohort (ovary), which never uses its own validation data: for mask_bal the
median of the λ chosen in the development training sets of the other trap cohorts (masking = 0), mapped to the nearest
grid value; for every other family the most frequent choice (ties: the setting closest to masking).
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


def _r8(name):
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


C = _r8("common")
CA = _r8("candidates")
DEV_TRAP_COHORTS = ("thyroid", "capsule", "isic")
ORDER = {"mask_bal": ["mask"] + [f"lam{x:g}" for x in CA.LAMBDAS], "mask_cmc": ["mask", *CA.CMC_MODES],
         "full_cmc": ["mask", *CA.CMC_MODES], "locrand": ["mask", "locrand"], "locrand_loc": ["mask", "locrand_loc"],
         "afr": ["mask"] + [f"g{x:g}" for x in CA.AFR_GAMMAS], "mask_cfs": ["mask"] + [f"mu{x:g}" for x in CA.CFS_MUS],
         "ens": ["mask"] + [f"a{x:g}" for x in CA.ENS_ALPHAS], "mask_cmc_uncontra": ["mask", *CA.CMC_MODES]}


def held_out_settings(ch: pd.DataFrame) -> dict:
    out = {}
    ch = ch[(ch.get("chosen") == True) & (ch.setting != "_plan")]  # noqa: E712
    for fam, order in ORDER.items():
        s = ch[ch.family == fam].setting
        if s.empty:
            continue
        if fam == "mask_bal":
            lam = s.map(lambda v: 0.0 if v == "mask" else float(v[3:]))
            med = float(np.median(lam))
            grid = [0.0, *CA.LAMBDAS]
            best = min(grid, key=lambda g: (abs(g - med), g))
            out[fam] = "mask" if best == 0.0 else f"lam{best:g}"
            out["mask_bal_median_lambda"] = med
        else:
            cnt = s.value_counts()
            top = cnt.max()
            out[fam] = next(o for o in order if cnt.get(o, 0) == top)
    return out


def git_rev(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(C.ROOT), "log", "-1", "--format=%H", "--", path], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = C.stage_dir("dev", a.smoke) / "trap"
    frames = []
    for coh in DEV_TRAP_COHORTS:
        f = dev / f"{coh}_dino518" / "choices.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        elif not a.smoke:
            raise SystemExit(f"missing development choices: {f}")
    ch = pd.concat(frames, ignore_index=True)
    dist = (ch[(ch.chosen == True) & (ch.setting != "_plan")]  # noqa: E712
            .groupby(["cohort", "family", "setting"]).size().rename("n").reset_index())
    checks = {}
    for f in sorted((C.out_dir(a.smoke) / "checks").glob("locrand_*.json")) if (C.out_dir(a.smoke) / "checks").exists() else []:
        checks[f.stem] = json.loads(f.read_text())
    fz = {"prereg": "docs/PREREGISTRATION_ROUND8.md", "prereg_commit": git_rev("docs/PREREGISTRATION_ROUND8.md"),
          "primary_candidates": list(C.PRIMARY), "selection": {"objective": "worst-group validation AUROC",
          "guard": "same-artifact validation AUROC >= masking head's - margin", "margin": C.GUARD_MARGIN,
          "split": "val_groups", "fallback": "masking head"},
          "grids": {k: v for k, v in ORDER.items()}, "seeds": {"dev": list(C.DEV_SEEDS), "confirm": list(C.CONF_SEEDS),
          "ft_traps": list(C.FT_TRAP_SEEDS)}, "held_out_cohort": "ovary",
          "held_out_settings": held_out_settings(ch), "dev_choice_counts": dist.to_dict("records"), "checks": checks}
    out = C.out_dir(a.smoke) / "frozen_choice.json"
    out.write_text(json.dumps(fz, indent=1, default=float))
    C.log(frozen=str(out), held_out=fz["held_out_settings"])


if __name__ == "__main__":
    main()
