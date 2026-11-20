#!/bin/bash
# A39 chain beside phase 5: (after the lesion U-Net) caches -> arm a features (5 bias-1 EPVT seeds) -> DINOv2 / U-MtE
# features -> arm d (bias 1) -> heads -> interim bias-1 comparison (reported; unadjusted) -> [bias-0.9 seeds by
# m2_p6_bias09.sh] -> bias-0.9 arm a / arm d -> heads both -> final comparison (Holm over 8). Every GPU step: at most one
# other GPU job, and never while phase 5's ISNet waits for the GPU (phase 5 has priority). Log: /workspace/logs/m2_p6_chain.log
export WTSS_DATA=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; PYW=/workspace/venv-wtss/bin/python; PYE=/workspace/envs/epvt/bin/python
cd /root/Where_shortcut_sits
gpu(){ local need=$1; shift; until ! ps -eo args | grep -q "[g]pu_wait.sh 15000 .*--net isnet"; do sleep 30; done; MAX_OTHER_JOBS=1 $Q/gpu_wait.sh $need "$@"; }
step(){ local n=$1; shift; echo "$(date -u +%F_%T) start $n"; "$@" || { echo "$(date -u +%F_%T) FAILED P6CHAIN $n"; exit 1; }; echo "$(date -u +%F_%T) done $n"; }
until grep -q "P6 prep done" /workspace/logs/m2_p6_prep.log; do grep -q "FAILED P6PREP" /workspace/logs/m2_p6_prep.log && exit 1; sleep 60; done
step cache $PYW $Q/p6_arms.py --stage cache --bias 1 0.9
step features_a_bias1 gpu 6000 $PYE $Q/p6_arms.py --stage features_a --bias 1
step features_bc gpu 6000 $PYW $Q/p6_arms.py --stage features_bc --bias 1 0.9
step armd_bias1 gpu 7000 $PYW $Q/p6_arms.py --stage armd --bias 1
step heads_bias1 $PYW $Q/p6_arms.py --stage heads --bias 1
step compare_bias1_interim $PYW $Q/p6_arms.py --stage compare --bias 1
echo "$(date -u +%F_%T) BIAS-1 COMPARISON READY"
until grep -q "P6 bias 0.9 seeds done" /workspace/logs/m2_p6_bias09.log 2>/dev/null; do grep -q "FAILED P6B09" /workspace/logs/m2_p6_bias09.log 2>/dev/null && exit 1; sleep 120; done
step features_a_bias09 gpu 6000 $PYE $Q/p6_arms.py --stage features_a --bias 0.9
step armd_bias09 gpu 7000 $PYW $Q/p6_arms.py --stage armd --bias 0.9
step heads_both $PYW $Q/p6_arms.py --stage heads --bias 1 0.9
step compare_final $PYW $Q/p6_arms.py --stage compare --bias 1 0.9 --final
echo "$(date -u +%F_%T) P6 A39 comparison done"
