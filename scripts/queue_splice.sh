#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
A="erm mask balanced splice mask_splice mte mte_protect mte_balanced"
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag splice --arms $A > logs/splice_thyroid.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag splice --arms $A > logs/splice_capsule.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_splice --arms $A > logs/splice_isic.log 2>&1
python scripts/run_drain_spec.py --backbone raddino518 --tag splice --arms $A > logs/splice_drain.log 2>&1
echo DONE >> logs/splice_drain.log
