#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_splice.sh" >/dev/null; do sleep 30; done
A="erm mask balanced mask_balanced mte mte_protect mte_balanced mte_protect_balanced jtt mask_jtt umte_jtt dfr mask_dfr"
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag auto --save_val --arms $A > logs/auto_thyroid.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag auto --save_val --arms $A > logs/auto_capsule.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_auto --save_val --arms $A > logs/auto_isic.log 2>&1
python scripts/run_drain_spec.py --backbone raddino518 --tag auto --save_val --arms $A > logs/auto_drain.log 2>&1
python scripts/analysis/adaptive_select.py results/thyroid/dino518_auto results/capsule/dino518_auto results/spec_e13/dino518_spec_auto results/cxr_drain/raddino518_auto > logs/auto_select.log 2>&1
echo DONE >> logs/auto_select.log
