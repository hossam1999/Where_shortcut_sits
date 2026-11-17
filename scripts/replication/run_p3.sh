#!/bin/bash
# One P3 (Bassi et al. 2024) training run with scripts/replication/p3_train.py in P3's environment.
#   run_p3.sh <name> <p3_train.py args...>      e.g. run_p3.sh isnet_corner_all_seed0 --model isnet --position corner --presence all
set -euo pipefail
name=$1; shift
D=${WTSS_DATA:-/workspace/data}; L=$D/replication/isnet/runs/$name; mkdir -p $L
cd /root/Where_shortcut_sits
WTSS_DATA=$D exec /workspace/envs/pahde/bin/python -u scripts/replication/p3_train.py --name $name "$@" > $L/stdout.log 2>&1
