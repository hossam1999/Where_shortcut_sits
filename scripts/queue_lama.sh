#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
for c in thyroid ovary capsule isic; do python scripts/data/precompute_lama.py --cohort $c > logs/lama_$c.log 2>&1; done
echo DONE >> logs/lama_isic.log
