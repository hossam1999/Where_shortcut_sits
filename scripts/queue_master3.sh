#!/bin/bash
# Strictly sequential: wait for running jobs, then CXR traps per backbone.
cd /root/wtss
source /venv/main/bin/activate
[ -f /root/.config/wtss/secrets.env ] && source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "e12_contrast_trap.py|run_leakage_experiment.py" >/dev/null; do sleep 30; done
for bb in raddino518 medsiglip448 dino518; do
  echo "=== $(date +%T) cxr $bb"; python scripts/run_cxr_traps.py --backbone $bb > logs/cxr_traps_$bb.log 2>&1; echo "rc=$?"
done
echo "=== DONE"
