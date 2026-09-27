#!/bin/bash
# docs/PREREGISTRATION_REVIEW2.md — ultrasound and capsule cohorts (R0 repro, R1 matched, R3 natural, R2 transplant)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src TQDM_DISABLE=1
t(){ echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for c in thyroid ovary capsule; do t python scripts/run_thyroid_traps.py --cohort $c --tag repro --arms erm mask; done
for c in thyroid ovary capsule; do t python scripts/run_thyroid_traps.py --cohort $c --tag matched --match --arms erm mask; done
for c in thyroid capsule; do t python scripts/run_natural.py --cohort $c --tag repro --save_val; done
for c in thyroid ovary capsule; do t python scripts/run_transplant.py --cohort $c; done
echo "=== $(date +%T) QUEUE US DONE"
