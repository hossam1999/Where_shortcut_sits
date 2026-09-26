#!/bin/bash
# Remaining synthetic replications E2-E5 (+ I2E arms), sequential, after queue_e15_clip.sh.
cd /root/wtss
source /venv/main/bin/activate
[ -f /root/.config/wtss/secrets.env ] && source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_e15_clip.sh" >/dev/null; do sleep 30; done
run(){ echo "=== $(date +%T) $1"; shift; "$@"; echo "rc=$?"; }
run E2_dermlip python scripts/run_synthetic.py --cohort isic2018 --backbone dermlip224 --tag main --workers 4 --arms erm mask balanced dfr leace --proposed > logs/synth_isic_dermlip.log 2>&1
run E2_dino224 python scripts/run_synthetic.py --cohort isic2018 --backbone dino224 --tag main --workers 4 --arms erm mask balanced dfr leace --proposed > logs/synth_isic_dino224.log 2>&1
run E5_dilation python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag dilation --overlaps 0.5 0.75 1.0 --workers 4 --arms erm dilate0 dilate10 dilate25 dilate50 > logs/synth_isic_dilation.log 2>&1
run E4_occlusion python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag occlusion --phase occlusion --workers 4 --arms erm mask inpaint > logs/synth_isic_occlusion.log 2>&1
run E3_stress python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --artifact ruler_variable --tag stress --overlaps 0 0.5 1 --workers 4 --arms erm mask balanced leace --proposed > logs/synth_isic_stress.log 2>&1
run leakage python scripts/run_leakage_experiment.py > logs/leakage.log 2>&1
echo "=== DONE"
