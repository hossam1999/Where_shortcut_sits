#!/usr/bin/env bash
# Part A (A1 addendum, docs/PREREGISTRATION_FINAL.md): finish the chest-drain and chest-device regenerations, their
# old-vs-new tables, and the retrospective theory comparison with the chest-radiograph cells (A3). Commits and pushes
# after each analysis. Log: results/rerun_2026-09-28/_logs/finish_A.log
#   bash scripts/release/finish_A.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source /venv/main/bin/activate
export PYTHONPATH="$PWD/src" WTSS_DATA=/root/data WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1
RR=results/rerun_2026-09-28
BR=claude/festive-bohr-95evo5
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0192Mq7C8i7t15VkZPfVxwbx"
log(){ echo "=== $(date +%T) $*"; }
commit(){ git add "$@" 2>/dev/null; git commit -q -m "$MSG

$TRAILER" && git push -q origin "$BR" && log "pushed $(git log --oneline -1)"; }

# ---- A1 3/4: chest drains ------------------------------------------------------------------------------------------
log "waiting for the DINOv2 drain run"
while pgrep -f "run_drain_spec.py --backbone dino[5]18" >/dev/null; do sleep 20; done
python scripts/verify/archived_rerun_compare.py --only capsule_tmpl,lama,drain
MSG="A1 addendum (3/4): real chest drains regenerated (RAD-DINO, DINOv2; incl. DFR) with per-image predictions; old vs new

Inputs rebuilt with the archived scripts: drain detector held-out precision 0.9501 at the registered threshold
(archived 0.95), NEATX CV AUROC 0.9955; drain cohort 29,687 images (archived 29,687); lung masks on the 4,076 images
shared with the archived nih_ptx cache: IoU mean 1.0000 (min 0.9998)."
commit scripts/verify/archived_rerun_compare.py results/bootstrap_correction/archived_old_vs_new.csv \
  results/bootstrap_correction/archived_old_vs_new.md $RR/cxr_drain

# ---- A1 4/4: chest devices -----------------------------------------------------------------------------------------
log "waiting for the device chain (cxr_rerun2.log: CXR DONE)"
until grep -q "CXR DONE" $RR/_logs/cxr_rerun2.log; do sleep 30; done
python scripts/verify/archived_rerun_compare.py --only capsule_tmpl,lama,drain,devices

# ---- A3: retrospective theory comparison with the chest-radiograph cells (separate folder; the regenerated theory
#      folder cited by the paper and the prospective 84-cell test of round 4 are not touched) -------------------------
cp -r $RR/theory /tmp/claude-0/theory_backup
WTSS_RESULTS=$PWD/$RR python scripts/analysis/theory_predict.py > $RR/_logs/theory_with_cxr.log 2>&1
rm -rf $RR/theory_with_cxr && mv $RR/theory $RR/theory_with_cxr && cp -r /tmp/claude-0/theory_backup $RR/theory
git checkout -- paper/ 2>/dev/null; git clean -fdq paper/ 2>/dev/null
python - <<'PY'
import json
n = json.load(open("results/rerun_2026-09-28/theory_with_cxr/prediction_summary.json"))
o = json.load(open("results/rerun_2026-09-28/theory/prediction_summary.json"))
for tag, s in (("without chest X-ray cells (regenerated, as cited)", o), ("with chest X-ray cells (retrospective)", n)):
    t1, t2 = s["T1_primary"]["theory"], s["T2_crossover"]
    print(f"{tag}: T1 n={t1['n']} r={t1['r']:.3f} MAE={t1['mae']:.3f} (ref clean {s['T1_primary']['ref_clean']['mae']:.3f}); "
          f"T2 signs {t2['sign_agree']}/{t2['n']} r={t2['r']:.3f}; all signs agree: {t2['sign_agree'] == t2['n']}; verdict: {s['verdict']}")
PY
MSG="A1 addendum (4/4): real chest-radiograph devices (RANZCR-CLiP linked to NIH) regenerated; old vs new; retrospective
theory comparison with the chest-radiograph cells (theory_with_cxr; prospective test unchanged)

Input check: RANZCR-to-NIH linkage 28,786 of 30,083 images (archived 28,786)."
commit scripts/verify/archived_rerun_compare.py results/bootstrap_correction/archived_old_vs_new.csv \
  results/bootstrap_correction/archived_old_vs_new.md $RR/cxr_traps $RR/theory_with_cxr
log "PART A DONE"
