#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "queue_lama.sh" >/dev/null; do sleep 60; done
python scripts/run_finetune_spec.py --cohort thyroid --arch vit_small_patch16_224.augreg_in21k_ft_in1k --lr 3e-5 > logs/ft_thyroid_vit.log 2>&1
