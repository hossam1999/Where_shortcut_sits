#!/usr/bin/env bash
# Round 8, Phase 1 (docs/ROUND8_DESIGN.md): scoreboard of existing results, criterion check, pair decomposition and the
# generated tables of the design document. Existing saved predictions only; no model is trained, no test set is scored.
#   bash scripts/round8/phase1.sh          # full (resumable: finished scoreboard parts are reused)
#   bash scripts/round8/phase1.sh smoke    # small subset -> results/round8/_smoke/ (git-ignored)
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=""; [ "${1:-}" = smoke ] && S="--smoke"
mkdir -p results/round8/logs
FAILED=()
run() { local n=$1; shift; echo "=== $(date '+%F %T') $n"; "$@" 2>&1 | tee "results/round8/logs/$n${S:+_smoke}.log"; [ "${PIPESTATUS[0]}" = 0 ] || FAILED+=("$n"); }
run scoreboard python scripts/round8/scoreboard.py $S --jobs 14
run decomposition python scripts/round8/pair_decomposition.py $S
if [ -z "$S" ]; then
  run criterion python scripts/round8/criterion_check.py
  run tables python scripts/round8/design_tables.py --doc
fi
echo "=== $(date '+%F %T') finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STEPS: ${FAILED[*]}"; exit 1; fi
echo "ALL STEPS OK"
