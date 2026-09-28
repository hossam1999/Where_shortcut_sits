#!/usr/bin/env bash
# Round 6, exploratory (NOT registered; decided after the registered results were known). CPU only: the thyroid
# features are copied from existing caches and the natural-split predictions are already saved.
#   bash scripts/round6/exploratory.sh
# E1: thyroid traps with the consensus rule (an image is removed only when BUSClean and MedGemma agree with each
#     other against our cell). E2: FT3 in the malignant in-ROI-caliper nodules that both labellers confirm.
# Output: results/round6/exploratory/ (never read by the registered tables).
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WTSS_DATA:?set WTSS_DATA to the data root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" WTSS_BOOTSTRAP=crossed TQDM_DISABLE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/round6/logs
python scripts/round6/clean_traps.py --cohort thyroid --rule consensus 2>&1 | tee results/round6/logs/E1_consensus.log
python scripts/round6/ft_thyroid.py --exploratory 2>&1 | tee results/round6/logs/E2_ft3_confirmed.log
git add results/round6/exploratory
git status --short results/round6/exploratory
