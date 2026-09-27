#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_protect.sh" >/dev/null; do sleep 30; done
A="erm mask dfr mte mte_balanced mask_dfr mte_dfr"
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag u10 --arms $A > logs/u10_thyroid.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag u10 --arms $A > logs/u10_capsule.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_u10 --arms $A > logs/u10_isic.log 2>&1
echo DONE >> logs/u10_isic.log
