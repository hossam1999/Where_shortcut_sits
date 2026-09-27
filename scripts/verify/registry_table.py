"""Registry of pre-registrations: first commit of each document vs first commit of its outcome results.

Commit times are set by the committing machine; the table documents order, not an external timestamp.
  python scripts/verify/registry_table.py        -> results/review2/registry.csv
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# registration -> result paths whose first commit carries outcome results (counts-only commits are flagged below)
RESULTS = {
    "PREREGISTRATION_ISIC2019_TRAPS.md": ["results/spec_e13"],
    "PRECOMMIT_MATCHED_TRAPS.md": ["results/spec_e13"],
    "PREREGISTRATION_CXR_DEVICE_TRAPS.md": ["results/cxr_traps"],
    "PREREGISTRATION_UNIVERSAL_I2E.md": ["results/thyroid/dino518_universal", "results/spec_e13/dino518_spec_universal"],
    "PREREGISTRATION_THYROID_TRAPS.md": ["results/thyroid"],
    "PREREGISTRATION_SLAS.md": ["results/slas"],
    "PREREGISTRATION_CAPSULE_TRAPS.md": ["results/capsule"],
    "PREREGISTRATION_CXR_DRAIN.md": ["results/cxr_drain"],
    "PREREGISTRATION_FINETUNE.md": ["results/finetune"],
    "PREREGISTRATION_OVARY_TRAPS.md": ["results/ovary"],
    "STATISTICAL_PLAN.md": ["results/PRIMARY_CLAIMS.md"],
    "PREREGISTRATION_NATURAL.md": ["results/natural"],
    "PREREGISTRATION_INPAINT_LAMA.md": ["results/thyroid/dino518_lama", "results/lama_comparison.json"],
    "PREREGISTRATION_TEXT_PROMPT.md": ["results/zero_shot"],
    "PREREGISTRATION_THEORY_PREDICTION.md": ["results/theory/prediction_summary.json"],
    "PREREGISTRATION_EMBEDDING_GROUPS.md": ["results/leakage/embedding_groups_compare.csv"],
    "PREREGISTRATION_SCALE.md": ["results/scale"],
    "PREREGISTRATION_UMTE_ABLATION.md": ["results/ablation_umte"],
    "PREREGISTRATION_FT_ERASE.md": "pickaxe:mte_post",  # new arms in an existing folder: first commit adding the arm
    "PREREGISTRATION_FT_UMTE.md": "pickaxe:umte_ft",
    "PREREGISTRATION_REVIEW2.md": ["results/review2/repro", "results/review2/matched", "results/review2/transplant",
                                   "results/review2/operating_points.csv"],
}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def first_commit(*paths, diff_filter="A") -> tuple[str, str]:
    out = git("log", "--reverse", f"--diff-filter={diff_filter}", "--format=%h|%aI", "--", *paths)
    return tuple(out.splitlines()[0].split("|")) if out else ("", "")


def main():
    rows = []
    for doc, res in RESULTS.items():
        h, t = first_commit(f"docs/{doc}")
        if isinstance(res, str):  # pickaxe: first commit whose results/ diff adds the arm name
            out = git("log", "--reverse", "--format=%h|%aI", f"-S{res.split(':', 1)[1]}", "--", "results/")
            rh, rt = out.splitlines()[0].split("|") if out else ("", "")
            rows.append({"registration": doc, "registered_commit": h, "registered_at": t, "first_outcome_commit": rh,
                         "first_outcome_at": rt, "order": "registration first" if rh and (t, h) < (rt, rh) else "check"})
            continue
        # first commit that adds a result file other than counts.csv (outcome-free cell counts)
        log = git("log", "--reverse", "--diff-filter=A", "--name-only", "--format=@%h|%aI", "--", *res)
        rh = rt = ""
        cur = None
        for line in log.splitlines():
            if line.startswith("@"):
                cur = line[1:].split("|")
            elif line.strip() and not line.endswith("counts.csv") and cur:
                rh, rt = cur
                break
        rows.append({"registration": doc, "registered_commit": h, "registered_at": t,
                     "first_outcome_commit": rh, "first_outcome_at": rt,
                     "order": "" if not rh else ("registration first" if (t, h) < (rt, rh) and h != rh else
                                                ("same commit" if h == rh else "RESULTS FIRST"))})
    d = pd.DataFrame(rows)
    out = ROOT / "results" / "review2"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "registry.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
