#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
A="erm mask balanced mte mte_protect mte_balanced"
for c in thyroid ovary; do
  python scripts/run_thyroid_traps.py --cohort $c --backbone dino518 --generic --tag sens_px50 --min_px 50 --arms $A > logs/sens_${c}_px50.log 2>&1
  python scripts/run_thyroid_traps.py --cohort $c --backbone dino518 --generic --tag sens_strictloc --rA 0.7 --rB 0.05 --arms $A > logs/sens_${c}_strictloc.log 2>&1
done
echo DONE >> logs/sens_ovary_strictloc.log
