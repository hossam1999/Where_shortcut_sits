#!/usr/bin/env bash
# Round 9 (docs/PREREGISTRATION_ROUND9.md): confirmation on seeds 9101-9505, then the analysis.
#   bash scripts/round9/run_all.sh [smoke]
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=""; L=results/round9/logs
if [ "${1:-}" = smoke ]; then S="--smoke"; L=results/round9/_smoke/logs; fi
mkdir -p "$L"; FAILED=(); P=scripts/round9
run() { local n=$1; shift; echo "=== $(date '+%F %T') $n"; "$@" 2>&1 | tee "$L/$n.log"; [ "${PIPESTATUS[0]}" = 0 ] || FAILED+=("$n"); }
run pytest python -m pytest -q tests
if [ ${#FAILED[@]} -gt 0 ]; then echo "STOPPED: tests failed"; exit 1; fi
for c in thyroid capsule ovary isic; do run "trap_${c}_dino518" python $P/confirm.py --kind trap --cohort $c $S; done
for c in thyroid capsule isic_BCN isic_HAM isic_MSK isic2020; do run "natural_$c" python $P/confirm.py --kind natural --cohort $c --jobs 5 $S; done
for c in thyroid capsule ovary; do run "trap_${c}_medsiglip448" python $P/confirm.py --kind trap --cohort $c --encoder medsiglip448 $S; done
for c in thyroid capsule; do run "trap_${c}_convnext384" python $P/confirm.py --kind trap --cohort $c --encoder convnext384 $S; done
run analyse python $P/analyse.py $S
echo "=== $(date '+%F %T') finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STEPS: ${FAILED[*]}"; exit 1; fi
echo "ALL STEPS OK"
