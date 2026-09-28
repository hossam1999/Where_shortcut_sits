#!/usr/bin/env bash
# Round 6, after the author's blinded rating (A4 / A4b): score the rating, then re-run the steps it can change.
#   bash scripts/round6/after_rating.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/round6/analyse_rating.py
# a model labeller that could only be judged against the rating (Amendment 2) may now count as informative
for c in thyroid ovary; do
  rm -f "results/round6/agreement/$c/agreement.json" "results/round6/clean_traps/$c/crossover.json"
  python scripts/round6/label_agreement.py --cohort "$c"
  python scripts/round6/clean_traps.py --cohort "$c"
done
rm -f results/round6/clean_traps/SUMMARY.csv results/round6/bias.csv
python scripts/round6/clean_traps.py --summary
python scripts/round6/bias_analysis.py
python scripts/round6/summarise.py
