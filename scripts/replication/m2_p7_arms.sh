#!/bin/bash
Q=/root/Where_shortcut_sits/scripts/replication; export WTSS_DATA=/workspace/data
cd /workspace/third_party/rrclarc
echo "$(date -u +%F_%T) start P7 arms features"
MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 6000 /workspace/envs/r2r/bin/python $Q/p7_arms.py --stage features || { echo "FAILED P7 arms features"; exit 1; }
echo "$(date -u +%F_%T) start P7 arms heads"
/workspace/venv-wtss/bin/python $Q/p7_arms.py --stage heads || { echo "FAILED P7 arms heads"; exit 1; }
echo "$(date -u +%F_%T) P7 arms done"
