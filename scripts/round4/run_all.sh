#!/usr/bin/env bash
# Round 4 end to end (docs/PREREGISTRATION_ROUND4.md). Resumable: finished steps are skipped on a re-run.
# GPU extraction runs in a child process; head fitting and the 10,000-replicate bootstrap then use every CPU.
# Seeds, replicate counts and copied feature rows are unchanged. New views are encoded at batch size 128.
#   bash scripts/round4/run_all.sh smoke     # quick check of every script on small subsets (results/round4/_smoke/)
#   bash scripts/round4/run_all.sh           # the full pre-registered analyses
# A failing step is reported and the remaining steps still run; the list of failures is printed at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root (the directory that contains isic2019/, us/, capsule/, ovary/, external/)}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=""; [ "${1:-}" = "smoke" ] && S="--smoke"
LOG=results/round4/logs${S:+/smoke}; mkdir -p "$LOG"
FAILED=()
run() {
  local name=$1; shift
  echo "=== $(date '+%F %T') $name"
  if python "$@" 2>&1 | tee "$LOG/$name.log"; then :; else echo "!!! $name FAILED -> $LOG/$name.log"; FAILED+=("$name"); fi
}

for c in ovary capsule thyroid isic; do run "R8_$c" scripts/round4/masking_variants.py --cohort "$c" $S; done
run R8_summary scripts/round4/masking_variants.py --summary $S

for c in ovary capsule thyroid isic; do run "R9_counts_$c" scripts/round4/dose_response.py --cohort "$c" --counts_only $S; done
if [ -z "$S" ]; then  # the bin counts are committed before any dose-response model is fitted
  { git add results/round4/dose_response/*/counts.csv results/round4/dose_response/*/bins_final.json &&
    git commit -q -m "Round 4 R9: bin counts and gate outcome, committed before any dose-response model is fitted" &&
    echo "=== R9 counts committed"; } || echo "!!! could not commit the R9 counts (git user not configured?)"
fi
for c in ovary capsule thyroid isic; do run "R9_$c" scripts/round4/dose_response.py --cohort "$c" $S; done
run R9_summary scripts/round4/dose_response.py --summary $S

run R10_prepare scripts/round4/natural_isic2020.py --prepare $S
run R10 scripts/round4/natural_isic2020.py $S

[ -z "$S" ] && run R11 scripts/round4/theory_round4.py
echo "=== $(date '+%F %T') finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STEPS: ${FAILED[*]}"; exit 1; fi
echo "ALL STEPS OK"
