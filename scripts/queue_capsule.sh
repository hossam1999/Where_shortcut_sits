#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "run_thyroid_traps.py --backbone medsiglip448|queue_ablation.sh" >/dev/null; do sleep 30; done
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 > logs/capsule_dino518.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag universal --arms erm mask balanced dfr i2e i2e_balanced mte mte_balanced mte_aug insert_aug > logs/capsule_dino518_universal.log 2>&1
echo DONE >> logs/capsule_dino518_universal.log
python scripts/run_synthetic.py --cohort capsule --backbone dino518 --artifact debris --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_capsule_dino518.log 2>&1
echo "rc=$?" >> logs/synth_capsule_dino518.log
