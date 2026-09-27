#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_medsiglip_generic.sh" >/dev/null; do sleep 60; done
A="erm mask balanced dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced mask_dfr mte_dfr"
python scripts/run_thyroid_traps.py --backbone convnext384 --generic --tag universal --arms $A > logs/convnext_thyroid_universal.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone convnext384 --generic --tag universal --arms $A > logs/convnext_capsule_universal.log 2>&1
echo DONE >> logs/convnext_capsule_universal.log
