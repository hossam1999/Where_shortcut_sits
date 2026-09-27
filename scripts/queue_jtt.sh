#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
A="erm mask mte jtt mask_jtt umte_jtt"
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag jtt --arms $A > logs/jtt_thyroid.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag jtt --arms $A > logs/jtt_capsule.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_jtt --arms $A > logs/jtt_isic.log 2>&1
echo DONE >> logs/jtt_isic.log
