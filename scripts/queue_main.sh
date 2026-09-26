#!/bin/bash
# Sequential (one heavy job at a time). Priority: thyroid (new real-artifact modality) > DermLIP items > CXR controlled.
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "python scripts/run_(cxr_traps|synthetic)" >/dev/null; do sleep 30; done
run(){ echo "=== $(date +%T) $1"; shift; "$@"; echo "rc=$?"; }
run thyroid_dino       python scripts/run_thyroid_traps.py --backbone dino518 > logs/thyroid_dino518.log 2>&1
run thyroid_dino_univ  python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag universal > logs/thyroid_dino518_universal.log 2>&1
run thyroid_medsiglip  python scripts/run_thyroid_traps.py --backbone medsiglip448 > logs/thyroid_medsiglip.log 2>&1
run E15_generic        python scripts/run_spec_e13.py --backbone dermlip224 --generic --tag spec_universal --arms erm mask balanced dfr i2e i2e_balanced mte mte_balanced > logs/spec_universal_dermlip.log 2>&1
run E2_dermlip         python scripts/run_synthetic.py --cohort isic2018 --backbone dermlip224 --tag main --workers 4 --arms erm mask balanced dfr leace --proposed > logs/synth_isic_dermlip.log 2>&1
run CXR_synth_raddino  python scripts/run_synthetic.py --cohort nih_ptx --backbone raddino518 --artifact tube --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_cxr_raddino518.log 2>&1
echo "=== DONE"
