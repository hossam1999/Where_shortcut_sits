#!/bin/bash
# CXR traps: primary (pre-registered) then device-matched follow-up, per backbone. Sequential.
cd /root/wtss
source /venv/main/bin/activate
[ -f /root/.config/wtss/secrets.env ] && source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "python scripts/run_cxr_traps.py" >/dev/null; do sleep 30; done
for bb in raddino518 medsiglip448 dino518; do
  echo "=== $(date +%T) cxr $bb primary"; python scripts/run_cxr_traps.py --backbone $bb >> logs/cxr_traps_$bb.log 2>&1; echo "rc=$?"
  echo "=== $(date +%T) cxr $bb device-matched"; python scripts/run_cxr_traps.py --backbone $bb --device_matched > logs/cxr_traps_${bb}_devmatched.log 2>&1; echo "rc=$?"
done
echo "=== DONE"
