#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_synth_protect.sh" >/dev/null; do sleep 30; done
python scripts/run_synthetic.py --cohort thyroid --backbone medsiglip448 --artifact caliper --tag protect --workers 4 --arms erm mask balanced --proposed > logs/synth_thyroid_medsiglip448_protect.log 2>&1
echo DONE >> logs/synth_thyroid_medsiglip448_protect.log
