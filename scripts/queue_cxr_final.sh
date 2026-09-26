#!/bin/bash
# Only the fair X-ray check remains: device-matched traps on RAD-DINO (after the running primary finishes).
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
while pgrep -f "python scripts/run_cxr_traps.py" >/dev/null; do sleep 30; done
echo "=== $(date +%T) raddino devmatched"; python scripts/run_cxr_traps.py --backbone raddino518 --device_matched > logs/cxr_traps_raddino518_devmatched.log 2>&1; echo "rc=$?"
echo "=== DONE"
