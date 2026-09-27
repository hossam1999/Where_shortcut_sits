#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
until grep -q DONE logs/lama_isic.log 2>/dev/null; do sleep 60; done
python scripts/run_spec_e13.py --backbone dino518 --lama --tag spec_lama --arms erm mask > logs/lama_trap_isic.log 2>&1
echo DONE >> logs/lama_trap_isic.log
