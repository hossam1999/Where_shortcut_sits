#!/bin/bash
# E15 (DermLIP, author protocol) then the CXR CLiP cohort preparation. Sequential.
cd /root/wtss
source /venv/main/bin/activate
[ -f /root/.config/wtss/secrets.env ] && source /root/.config/wtss/secrets.env   # HF_TOKEN (gated DermLIP / MedSigLIP)
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
echo "=== $(date +%T) E15"; python scripts/run_spec_e13.py --backbone dermlip224 > logs/spec_e15_dermlip.log 2>&1; echo "rc=$?"
echo "=== $(date +%T) CLiP prep"; python scripts/data/prepare_clip.py > logs/prep_clip.log 2>&1; echo "rc=$?"
echo "=== $(date +%T) DONE"
