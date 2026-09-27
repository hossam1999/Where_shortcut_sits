#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
python scripts/run_natural.py --cohort capsule > logs/natural_capsule.log 2>&1
for s in HAM BCN MSK; do python scripts/run_natural.py --cohort isic_$s > logs/natural_isic_$s.log 2>&1; done
echo DONE >> logs/natural_isic_MSK.log
