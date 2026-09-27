#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_synth_medsiglip.sh" >/dev/null; do sleep 30; done
for bb in medsiglip448 dino518; do
  python scripts/run_synthetic.py --cohort capsule --backbone $bb --artifact debris --tag protect --workers 4 --arms erm mask balanced --proposed > logs/synth_capsule_${bb}_protect.log 2>&1
done
echo DONE >> logs/synth_capsule_dino518_protect.log
