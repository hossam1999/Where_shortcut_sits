#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "bash scripts/queue_derm" >/dev/null; do sleep 30; done
echo "=== $(date +%T) E15 (cached features, parallel)"; python scripts/run_spec_e13.py --backbone dermlip224 > logs/spec_e15_dermlip.log 2>&1; echo "rc=$?"
echo "=== DONE"
