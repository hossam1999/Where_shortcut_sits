#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
python scripts/run_synthetic.py --cohort capsule --backbone medsiglip448 --artifact debris --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_capsule_medsiglip.log 2>&1
echo "rc=$?" >> logs/synth_capsule_medsiglip.log
python scripts/run_synthetic.py --cohort thyroid --backbone medsiglip448 --artifact caliper --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_thyroid_medsiglip.log 2>&1
echo "rc=$?" >> logs/synth_thyroid_medsiglip.log
