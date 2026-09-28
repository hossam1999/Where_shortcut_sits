"""Post-processing lanes (new file; scripts/final/tasks.py is in use by running lanes and is not edited).

fix:  ISIC copy tasks re-queued with metrics_per_seed.csv (run_spec_e13.py reads it; per-seed AUROCs, no intervals).
post: analyses that need the regenerated runs (pilot ledger, mechanism, primary family, figures, theory, natural
      clinical metrics, crossed operating points, second/third-review summaries).
"""
from __future__ import annotations

from tasks import A_SCALE, NEW, OLD, T, spec  # noqa: F401

FIX = []
for tag, extra in (("repro", ()), ("matched", ("--match",)), ("matched05", ("--match", "--match_caliper", "0.05"))):
    FIX.append(spec(f"fix_isic_{tag}", "dino518", tag, ["erm", "mask"], *extra,
                    copy=[f"spec_e13/dino518_{tag}/predictions.csv.gz", f"spec_e13/dino518_{tag}/metrics_per_seed.csv"]))
for b in ("dinos518", "dinol518"):
    FIX.append(spec(f"fix_isic_scale_{b}", b, "spec_scale", "erm mask balanced dfr mte mte_balanced mte_protect mte_aug".split(),
                    "--generic", copy=[f"spec_e13/{b}_spec_scale/predictions.csv.gz", f"spec_e13/{b}_spec_scale/metrics_per_seed.csv"]))

POST = [
    T("a2_matched_regression", "scripts/analysis/matched_regression.py"),
    T("e7_e14", "scripts/analysis/e7_e14.py", wait_for=str(NEW / "_done" / "t_isic_spec.ok"), wait_hours=12),
    T("e10_hair", "scripts/analysis/e10_hair.py"),
    T("e12_contrast", "scripts/analysis/e12_contrast_trap.py"),
    T("op_crossed_all", "scripts/final/op_crossed_all.py"),
    T("clinical_repro", "scripts/analysis/clinical_metrics.py", "--suffix", "repro"),
    T("review2_summary", "scripts/analysis/review2_summary.py"),
    T("crossed_ci", "scripts/analysis/crossed_ci.py"),
    T("literature", "scripts/analysis/literature_table.py"),
]
# after the trap lane has produced the universal / scale runs
POST_LATE = [
    T("primary_claims", "scripts/analysis/primary_claims.py", wait_for=str(NEW / "_done" / "u_isic_univ.ok"), wait_hours=14),
    T("scale_compare", "scripts/analysis/scale_compare.py", wait_for=str(NEW / "_done" / "scale_capsule_dinol518.ok"), wait_hours=14),
    T("cross_cohort", "scripts/make_cross_cohort_table.py"),
    T("umte_ablation", "scripts/analysis/umte_ablation.py"),
    T("summary_ledger", "scripts/make_summary.py", wait_for=str(NEW / "_done" / "syn_isic_dermlip224.ok"), wait_hours=14),
    T("mechanism_figure", "scripts/analysis/mechanism_figure.py"),
    T("theory_predict", "scripts/analysis/theory_predict.py"),
]
import tasks as _t  # noqa: E402
RETRY = [t for t in _t.TRAPS if t["name"] == "t_thy_med_main"]
POST_LATE2 = RETRY + [dict(t, wait_for=None) for t in POST_LATE] + [
    T("review2_summary_late", "scripts/analysis/review2_summary.py"),
    T("mechanism_crossed", "scripts/final/mechanism_crossed.py"),
    T("registry_final", "scripts/final/registry_final.py"),
]
LANES = {"fix": FIX, "post": FIX + POST, "post_late": POST_LATE, "late": POST_LATE2}
