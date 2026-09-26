#!/bin/bash
# Controlled overlap sweep in chest radiography (pneumothorax; lung ROI), after the running real-hair job.
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "run_spec_e13.py --backbone dino518 --generic" >/dev/null; do sleep 30; done
for bb in raddino518 dino518; do
  echo "=== $(date +%T) CXR synthetic tube $bb"
  python scripts/run_synthetic.py --cohort nih_ptx --backbone $bb --artifact tube --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_cxr_$bb.log 2>&1; echo "rc=$?"
done
echo "=== DONE"
