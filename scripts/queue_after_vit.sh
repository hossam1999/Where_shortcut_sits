#!/bin/bash
# Strictly sequential (the previous concurrent attempt was killed for memory). Waits on the ViT log, not on pgrep.
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
until grep -qE "seed_delta_mean|Traceback" logs/ft_thyroid_vit.log 2>/dev/null; do sleep 60; done
for s in HAM BCN MSK; do python scripts/run_natural.py --cohort isic_$s > logs/natural_isic_$s.log 2>&1; done
python scripts/run_natural.py --cohort capsule > logs/natural_capsule.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --lama --tag spec_lama --arms erm mask > logs/lama_trap_isic.log 2>&1
echo ALLDONE >> logs/queue_after_vit.log
