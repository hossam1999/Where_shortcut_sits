#!/bin/bash
# docs/PREREGISTRATION_REVIEW2.md — ISIC 2019 hair (R0 repro, R1 matched, R3 natural, R2 transplant, R4 fine-tuning),
# then the two amendments registered before this review (ISIC hair scale; ovary fine-tuning power extension).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src TQDM_DISABLE=1
t(){ echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
t python scripts/run_spec_e13.py --tag repro --arms erm mask
t python scripts/run_spec_e13.py --tag matched --match --arms erm mask
for s in BCN HAM MSK; do t python scripts/run_natural.py --cohort isic_$s --tag repro --save_val; done
t python scripts/run_transplant.py --cohort isic
t python scripts/run_finetune_spec.py --cohort isic --arms erm mask balanced mask_balanced
for b in dinos518 dinol518; do t python scripts/run_spec_e13.py --backbone $b --generic --tag spec_scale \
    --arms erm mask balanced dfr mte mte_balanced mte_protect mte_aug; done
for s in 42 123 456; do t python scripts/run_finetune_spec.py --cohort ovary --tag power --env_seed $s --arms mask umte_cons_ft --traps trapA; done
# amendment 1: fine-tuned ovary crossover with 15 clusters
for s in 42 123 456; do t python scripts/run_finetune_spec.py --cohort ovary --tag power --env_seed $s --arms erm mask; done
echo "=== $(date +%T) QUEUE ISIC DONE"
