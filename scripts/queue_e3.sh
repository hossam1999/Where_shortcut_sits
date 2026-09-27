#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_thyroid_synth.sh|queue_capsule.sh|queue_protect.sh|queue_main.sh" >/dev/null; do sleep 60; done
python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --artifact ruler_variable --tag stress --overlaps 0 0.5 1 --workers 4 --arms erm mask balanced > logs/synth_isic_stress.log 2>&1
echo "rc=$?" >> logs/synth_isic_stress.log
