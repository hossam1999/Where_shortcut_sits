#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_u10.sh" >/dev/null; do sleep 60; done
A="erm mask balanced dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced mte_dfr mask_dfr"
python scripts/run_thyroid_traps.py --backbone medsiglip448 --generic --tag universal --arms $A > logs/medsiglip_thyroid_universal.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone medsiglip448 > logs/medsiglip_capsule_main.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone medsiglip448 --generic --tag universal --arms $A > logs/medsiglip_capsule_universal.log 2>&1
echo DONE >> logs/medsiglip_capsule_universal.log
