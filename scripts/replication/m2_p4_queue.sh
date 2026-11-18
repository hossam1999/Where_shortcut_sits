#!/bin/bash
# Machine 2 (A27): P4 Segmentation-ERM splits 1-3, one after the other, same command as machine 1's short_queue.sh.
# Each run resumes from its own checkpoint.pt (machine 2's; machine 1's split-1 checkpoint is not used). A finished split
# (preds_test.csv present) is skipped.   nohup scripts/replication/m2_p4_queue.sh > /workspace/logs/m2_p4_queue.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication
mkdir -p $D/replication/saab/runs
for k in 1 2 3; do
  [ -s $D/replication/saab/runs/seg_split$k/preds_test.csv ] && { echo "$(date -u +%F_%T) seg split $k done, skipped"; continue; }
  echo "$(date -u +%F_%T) seg split $k start"
  $Q/gpu_wait.sh 8000 bash -c "/workspace/envs/saab/bin/python -u $Q/p4_train.py --method seg --split $k --grad_ckpt >> $D/replication/saab/seg_split$k.log 2>&1" \
    || { echo "$(date -u +%F_%T) seg split $k FAILED"; exit 1; }
  echo "$(date -u +%F_%T) seg split $k finished"
done
echo "$(date -u +%F_%T) queue done"
