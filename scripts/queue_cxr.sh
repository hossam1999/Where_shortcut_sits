#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
[ -f /root/.config/wtss/secrets.env ] && source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
for bb in raddino518 medsiglip448 dino518; do
  echo "=== $(date +%T) $bb"; python scripts/run_cxr_traps.py --backbone $bb > logs/cxr_traps_$bb.log 2>&1; echo "rc=$?"
done
echo "=== DONE"
