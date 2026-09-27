#!/bin/bash
# Sequential among themselves; runs alongside the ViT fine-tuning (2 jobs fit in memory; 3 did not).
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
for s in HAM BCN MSK; do python scripts/run_natural.py --cohort isic_$s > logs/natural_isic_$s.log 2>&1; done
python scripts/run_natural.py --cohort capsule > logs/natural_capsule.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --lama --tag spec_lama --arms erm mask > logs/lama_trap_isic.log 2>&1
echo ALLDONE >> logs/queue_after_vit.log
