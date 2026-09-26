#!/bin/bash
# Sequential GPU queue (one GPU job at a time avoids contention/OOM). Each step is idempotent:
# cached features/views are reused, so the queue can be re-run after an interruption.
set -x
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
L=logs
wait_for(){ while pgrep -f "$1" >/dev/null; do sleep 30; done; }
wait_for "run_synthetic.py --cohort isic2018 --backbone dino518 --tag main"
wait_for "train_lesion_unet.py --stage train"
python scripts/data/train_lesion_unet.py --stage predict > $L/unet_predict.log 2>&1
python scripts/data/prepare_isic2019.py --stage cache --workers 8 > $L/isic2019_cache.log 2>&1 &
CACHE_PID=$!
python scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag proposed --workers 3 --arms erm --proposed > $L/synth_isic_dino518_proposed.log 2>&1
python scripts/data/prepare_cxr.py --stage detector > $L/prep_cxr_detector.log 2>&1
wait $CACHE_PID
python scripts/run_traps.py --cohort isic2019 --counts_only > $L/traps_isic2019_counts.log 2>&1
echo QUEUE1_DONE
