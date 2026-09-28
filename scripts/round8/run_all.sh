#!/usr/bin/env bash
# Round 8 (docs/PREREGISTRATION_ROUND8.md). Order: tests -> locrand placements / pasted features / probe check ->
# DEVELOPMENT (validation only) -> freeze -> commit + push results/round8/frozen_choice.json -> CONFIRMATION -> analysis.
#   bash scripts/round8/run_all.sh          # full run (resumable: finished steps are skipped)
#   bash scripts/round8/run_all.sh smoke    # every script on small subsets; validation images stand in for every
#                                           # test set; output in results/round8/_smoke/ (git-ignored)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=""; L=results/round8/logs; BR=claude/festive-bohr-95evo5
if [ "${1:-}" = smoke ]; then S="--smoke"; L=results/round8/_smoke/logs; fi
mkdir -p "$L"
FAILED=()
run() { local n=$1; shift; echo "=== $(date '+%F %T') $n"; "$@" 2>&1 | tee "$L/$n.log"; [ "${PIPESTATUS[0]}" = 0 ] || FAILED+=("$n"); }
stop_if_failed() { if [ ${#FAILED[@]} -gt 0 ]; then echo "STOPPED after failures: ${FAILED[*]}"; exit 1; fi; }
P=scripts/round8

run pytest python -m pytest -q tests
stop_if_failed

# --- N3 inputs (no labels, no test scoring)
for c in thyroid capsule ovary isic2019; do
  run "locrand_placements_$c" python $P/locrand.py --cohort $c --step placements $S
  run "locrand_extract_$c" python $P/locrand.py --cohort $c --step extract $S
  run "locrand_probe_$c" python $P/locrand.py --cohort $c --step probe $S
done
stop_if_failed

# --- development: validation data only; the held-out cohort (ovary) is never run here
for c in thyroid capsule isic; do run "dev_trap_${c}_dino518" python $P/heads_run.py --stage dev --kind trap --cohort $c $S; done
for c in thyroid capsule; do
  run "dev_trap_${c}_medsiglip448" python $P/heads_run.py --stage dev --kind trap --cohort $c --encoder medsiglip448 $S
  run "dev_trap_${c}_convnext384" python $P/heads_run.py --stage dev --kind trap --cohort $c --encoder convnext384 $S
done
run freeze python $P/freeze.py $S
stop_if_failed

if [ -z "$S" ]; then  # the frozen choice is committed and pushed before any confirmation prediction exists
  git add results/round8/frozen_choice.json results/round8/dev results/round8/checks
  git commit -m "Round 8: frozen development choices (validation only), before any confirmation run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || { echo "commit of the frozen choice failed"; exit 1; }
  git push origin "$BR" || { echo "push of the frozen choice failed"; exit 1; }
fi

# --- confirmation
for c in thyroid capsule ovary isic; do run "conf_trap_${c}_dino518" python $P/heads_run.py --stage confirm --kind trap --cohort $c $S; done
for c in thyroid capsule ovary; do run "conf_trap_${c}_medsiglip448" python $P/heads_run.py --stage confirm --kind trap --cohort $c --encoder medsiglip448 $S; done
for c in thyroid capsule; do run "conf_trap_${c}_convnext384" python $P/heads_run.py --stage confirm --kind trap --cohort $c --encoder convnext384 $S; done
for c in thyroid capsule isic_BCN isic_HAM isic_MSK isic2020; do
  run "conf_natural_${c}" python $P/heads_run.py --stage confirm --kind natural --cohort $c --jobs 5 $S
done
for s in thyroid capsule ovary isic2018; do  # NIH chest tube: labels not on this machine (prereg section 2)
  run "conf_sweep_$s" python $P/sweeps.py --sweep $s $S; done
run conf_ft_natural python $P/ft.py --part natural $S
run conf_ft_traps python $P/ft.py --part traps $S
run analyse python $P/analyse.py $S

echo "=== $(date '+%F %T') finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STEPS: ${FAILED[*]}"; exit 1; fi
echo "ALL STEPS OK"
