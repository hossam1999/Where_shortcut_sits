#!/bin/bash
# Machine 2: P4 Seg-ERM with two splits at once (pairing kept only if measured total throughput rises; STATUS).
# Starts every split k in 1..3 that has no preds_test.csv and no running process, keeping at most 2 running.
#   nohup scripts/replication/m2_p4_pair.sh > /workspace/logs/m2_p4_pair.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication
running(){ ps -eo args | grep -c "^/workspace/envs/saab/bin/python -u $Q/p4_train.py --method seg --split $1 "; }
nrun(){ ps -eo args | grep "^/workspace/envs/saab/bin/python -u $Q/p4_train.py --method seg" | awk '{print $7}' | sort -u | wc -l; }
while :; do
  left=0
  for k in 1 2 3; do
    [ -s $D/replication/saab/runs/seg_split$k/preds_test.csv ] && continue
    left=1
    [ "$(running $k)" -gt 0 ] && continue
    [ "$(nrun)" -ge 2 ] && continue
    echo "$(date -u +%F_%T) start seg split $k"
    MAX_OTHER_JOBS=1 $Q/gpu_wait.sh 8000 bash -c "/workspace/envs/saab/bin/python -u $Q/p4_train.py --method seg --split $k --grad_ckpt >> $D/replication/saab/seg_split$k.log 2>&1" &
    sleep 120
  done
  [ $left -eq 0 ] && break
  sleep 60
done
wait; echo "$(date -u +%F_%T) all splits done"
