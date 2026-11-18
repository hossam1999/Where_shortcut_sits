#!/bin/bash
# Machine 2: after each P4 Seg-ERM split is scored -> arm a features (GPU, minutes, beside the running split) -> arm a heads
# and the P4 gate / arm-d comparison (CPU). Ends when all three splits are done.
#   nohup scripts/replication/m2_p4_post.sh > /workspace/logs/m2_p4_post.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; export WTSS_DATA=$D
P=${P4_RUN:-seg}; export P4_RUN=$P
done_k(){ [ -s $D/replication/saab/arm_a/${P}_split$1_encoder_masked.npz ]; }
while :; do
  for k in 1 2 3; do
    if [ -s $D/replication/saab/runs/${P}_split$k/preds_test.csv ] && ! done_k $k; then
      echo "$(date -u +%F_%T) split $k scored -> arm a features"
      MAX_OTHER_JOBS=1 $Q/gpu_wait.sh 4000 /workspace/envs/saab/bin/python -u $Q/p4_arm_a.py --stage features --split $k || echo "FAILED features $k"
      nice -n 10 /workspace/venv-wtss/bin/python $Q/p4_arm_a.py --stage heads > /workspace/logs/p4_arm_a_heads.log 2>&1 || echo "FAILED heads"
      nice -n 10 /workspace/venv-wtss/bin/python $Q/p4_gate.py > /workspace/logs/p4_gate.log 2>&1 || echo "FAILED gate"
      echo "$(date -u +%F_%T) split $k post-processing done"; tail -3 /workspace/logs/p4_gate.log
    fi
  done
  done_k 1 && done_k 2 && done_k 3 && break
  sleep 60
done
echo "$(date -u +%F_%T) all P4 post-processing done"
