#!/bin/bash
# docs/PREREGISTRATION_REVIEW3.md — R5 paste-edge control (neutral tissue through the transplant masks)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src TQDM_DISABLE=1
t(){ echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for c in ovary capsule isic thyroid; do t python scripts/run_transplant.py --cohort $c --neutral --arms erm mask; done
echo "=== $(date +%T) QUEUE R5 DONE"
