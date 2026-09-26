#!/bin/bash
# Real chest-drain pipeline: detector -> trap cohort/cache -> traps (RAD-DINO, DINOv2). Idempotent.
set -x
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
L=logs
python scripts/data/prepare_cxr.py --stage detector > $L/prep_cxr_detector.log 2>&1 || exit 1
python scripts/data/prepare_cxr.py --stage drain > $L/prep_cxr_drain.log 2>&1 || exit 1
python scripts/run_traps.py --cohort nih_drain --backbone raddino518 > $L/traps_drain_raddino.log 2>&1
python scripts/run_traps.py --cohort nih_drain --backbone dino518 > $L/traps_drain_dino.log 2>&1
echo QUEUE3_DONE
