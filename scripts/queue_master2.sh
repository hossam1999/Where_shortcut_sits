#!/bin/bash
# Master queue v2 (priority: DermLIP replication, then real chest drains, then CXR synthetic, then the rest).
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
L=logs
step(){ echo "=== $(date +%T) START $1"; shift; "$@"; echo "=== $(date +%T) END rc=$?"; }
while pgrep -f "prepare_cxr.py --stage detector" >/dev/null; do sleep 20; done
step traps_isic2019_dermlip  python scripts/run_traps.py --cohort isic2019 --backbone dermlip224 --workers 5 > $L/traps_isic2019_dermlip.log 2>&1
step traps_isic2019_dermlip_matched python scripts/run_traps.py --cohort isic2019 --matched --tag matched --backbone dermlip224 --workers 5 > $L/traps_isic2019_dermlip_matched.log 2>&1
[ -f /root/data/cxr/prepared/DRAIN_DETECTOR.json ] || step cxr_detector python scripts/data/prepare_cxr.py --stage detector > $L/prep_cxr_detector.log 2>&1
step cxr_drain_cohort        python scripts/data/prepare_cxr.py --stage drain > $L/prep_cxr_drain.log 2>&1
step traps_drain_raddino     python scripts/run_traps.py --cohort nih_drain --backbone raddino518 --workers 5 > $L/traps_drain_raddino.log 2>&1
step synth_cxr_raddino       python scripts/run_synthetic.py --cohort nih_ptx --backbone raddino518 --artifact tube --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > $L/synth_cxr_raddino.log 2>&1
step traps_drain_dino        python scripts/run_traps.py --cohort nih_drain --backbone dino518 --workers 5 > $L/traps_drain_dino.log 2>&1
step synth_cxr_dino          python scripts/run_synthetic.py --cohort nih_ptx --backbone dino518 --artifact tube --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > $L/synth_cxr_dino.log 2>&1
step synth_isic_dermlip      python scripts/run_synthetic.py --cohort isic2018 --backbone dermlip224 --tag main --workers 4 --arms erm mask balanced dfr leace --proposed > $L/synth_isic_dermlip.log 2>&1
step synth_isic_dino224      python scripts/run_synthetic.py --cohort isic2018 --backbone dino224 --tag main --workers 4 --arms erm mask balanced dfr leace --proposed > $L/synth_isic_dino224.log 2>&1
step synth_isic_stress       python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --artifact ruler_variable --tag stress --overlaps 0 0.5 1 --workers 4 --arms erm mask balanced leace --proposed > $L/synth_isic_stress.log 2>&1
step synth_isic_occlusion    python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag occlusion --phase occlusion --workers 4 --arms erm mask inpaint > $L/synth_isic_occlusion.log 2>&1
step leakage                 python scripts/run_leakage_experiment.py > $L/leakage.log 2>&1
echo "=== MASTER2_DONE"
