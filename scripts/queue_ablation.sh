#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag ablation --arms erm mask insert_aug mte mte_aug > logs/ablation_thyroid.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_ablation --arms erm mask insert_aug mte mte_aug > logs/ablation_isic.log 2>&1
echo DONE >> logs/ablation_isic.log
