#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while ! grep -q DONE logs/capsule_dino518_universal.log 2>/dev/null || pgrep -f "tag spec_pbal" >/dev/null; do sleep 30; done
A="erm mask mte mte_balanced mte_protect mte_protect_balanced"
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --tag protect_tmpl --arms $A > logs/protect_capsule_tmpl.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag protect_generic --arms $A > logs/protect_capsule_generic.log 2>&1
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag protect_generic --arms $A > logs/protect_thyroid_generic.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_protect_generic --arms $A > logs/protect_isic_generic.log 2>&1
echo DONE >> logs/protect_isic_generic.log
