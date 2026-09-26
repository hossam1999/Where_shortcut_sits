#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
[ -f /root/.config/wtss/secrets.env ] && source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_master3.sh" >/dev/null; do sleep 30; done
echo "=== $(date +%T) E3"; python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --artifact ruler_variable --tag stress --overlaps 0 0.5 1 --workers 4 --arms erm mask balanced leace --proposed > logs/synth_isic_stress.log 2>&1; echo "rc=$?"
echo "=== $(date +%T) E2 DermLIP"; python scripts/run_synthetic.py --cohort isic2018 --backbone dermlip224 --tag main --workers 4 --arms erm mask balanced dfr leace --proposed > logs/synth_isic_dermlip.log 2>&1; echo "rc=$?"
echo "=== $(date +%T) E15"; python scripts/run_spec_e13.py --backbone dermlip224 > logs/spec_e15_dermlip.log 2>&1; echo "rc=$?"
echo "=== DONE"
