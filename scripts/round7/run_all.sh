#!/usr/bin/env bash
# Round 7 (docs/PREREGISTRATION_ROUND7.md) and the exploratory round-6 analyses E1-E2. CPU only; saved predictions and
# cached features only.
#   bash scripts/round7/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/round7/logs
FAILED=()
run() { local n=$1; shift; echo "=== $(date '+%F %T') $n"; "$@" 2>&1 | tee "results/round7/logs/$n.log"; [ "${PIPESTATUS[0]}" = 0 ] || FAILED+=("$n"); }
run pytest python -m pytest -q tests
run smoke python scripts/round7/matched_thresholds.py --smoke
run TM python scripts/round7/matched_thresholds.py
run E1_E2 bash scripts/round6/exploratory.sh
git add results/round7/matched_thresholds.csv results/round7/tm.json results/round7/SUMMARY.md results/round6/exploratory 2>/dev/null
echo "=== $(date '+%F %T') finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STEPS: ${FAILED[*]}"; exit 1; fi
echo "ALL STEPS OK"
