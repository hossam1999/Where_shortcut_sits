#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_sens.sh" >/dev/null; do sleep 30; done
for c in thyroid ovary; do
  python scripts/run_thyroid_traps.py --cohort $c --backbone dino518 --lama --tag lama --arms erm mask > logs/lama_trap_$c.log 2>&1
done
echo DONE >> logs/lama_trap_ovary.log
