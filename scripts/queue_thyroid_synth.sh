#!/bin/bash
# Runs after queue_main.sh and the SLAS run finish (one heavy job at a time).
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_main.sh|scripts/run_slas.py|slas_fewshot" >/dev/null; do sleep 60; done
python scripts/run_synthetic.py --cohort thyroid --backbone dino518 --artifact caliper --tag main --workers 4 \
  --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_thyroid_dino518.log 2>&1
echo "rc=$?" >> logs/synth_thyroid_dino518.log
