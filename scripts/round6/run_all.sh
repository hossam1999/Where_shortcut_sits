#!/usr/bin/env bash
# Round 6 end to end (docs/PREREGISTRATION_ROUND6.md, including Amendment 1).
# Resumable: finished steps are skipped when their output files already exist.
#   bash scripts/round6/run_all.sh smoke
#   bash scripts/round6/run_all.sh
# A failing step is reported and the remaining steps still run.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=""; SMOKE=0
if [ "${1:-}" = "smoke" ]; then S="--smoke"; SMOKE=1; fi
ROOT=results/round6
if [ "$SMOKE" = 1 ]; then ROOT=results/round6/_smoke; fi
LOG=$ROOT/logs
mkdir -p "$LOG"
FAILED=()
run() {
  local name=$1; shift
  echo "=== $(date '+%F %T') $name"
  if python "$@" 2>&1 | tee "$LOG/$name.log"; then :; else echo "!!! $name FAILED -> $LOG/$name.log"; FAILED+=("$name"); fi
}
skip_if() {
  # skip_if <marker> <name> <cmd...>
  local marker=$1 name=$2; shift 2
  if [ -e "$marker" ]; then echo "=== skip $name ($marker exists)"; return 0; fi
  run "$name" "$@"
}

run pytest -m pytest -q tests

skip_if "$ROOT/coverage.csv" A0_sources scripts/round6/sources.py $S
if [ "$SMOKE" = 0 ] && [ -f results/round6/coverage.csv ]; then
  git add results/round6/sources.json results/round6/coverage.csv results/round6/coverage.json results/round6/matches &&
    git commit -q -m "Round 6 A0: source licences and coverage, committed before the remaining analyses." &&
    echo "=== A0 coverage committed" || echo "!!! could not commit the coverage report"
fi

skip_if "$ROOT/ft_recipe.json" B1_select scripts/round6/ft_thyroid.py --select $S
if [ "$SMOKE" = 0 ] && [ -f results/round6/ft_recipe.json ]; then
  git add results/round6/ft_recipe.json &&
    git commit -q -m "Round 6 B1: validation-only fine-tune recipe, committed before any final model." &&
    echo "=== ft_recipe committed" || echo "!!! could not commit ft_recipe.json"
fi

for c in isic thyroid capsule ovary; do
  skip_if "$ROOT/agreement/$c/agreement.json" "A1_$c" scripts/round6/label_agreement.py --cohort "$c" $S
done
for c in isic thyroid capsule ovary; do
  skip_if "$ROOT/clean_traps/$c/crossover.json" "A2_$c" scripts/round6/clean_traps.py --cohort "$c" $S
done
skip_if "$ROOT/clean_traps/SUMMARY.csv" A2_summary scripts/round6/clean_traps.py --summary $S
skip_if "$ROOT/bias.csv" A3_bias scripts/round6/bias_analysis.py $S

run A4_rating_check scripts/round6/rating_app.py --pass 1 $S
run A4_medgemma scripts/round6/medgemma_audit.py $S
run A4_analyse scripts/round6/analyse_rating.py $S
skip_if "$ROOT/a4b_KEY_SHA256.txt" A4b scripts/round6/a4b_sample.py $S

skip_if "$ROOT/ft_natural/predictions.csv.gz" B2_natural scripts/round6/ft_thyroid.py --natural $S
run B2_traps scripts/round6/ft_thyroid.py --traps $S
run B2_analyse scripts/round6/ft_thyroid.py --analyse $S
run summary scripts/round6/summarise.py $S

echo "=== $(date '+%F %T') finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STEPS: ${FAILED[*]}"; exit 1; fi
echo "ALL STEPS OK"
echo "Rating tool (after the analyses, in another session):"
echo "  python scripts/round6/rating_app.py --pass 1"
echo "  ssh -p \${VAST_TCP_PORT_22} -L 8765:127.0.0.1:8765 root@\${PUBLIC_IPADDR}"
echo "  then open http://localhost:8765"
