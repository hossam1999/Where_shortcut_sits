#!/bin/bash
# Short jobs, one at a time beside the current long job (scheduling rules 2-4). Measured 2026-11-17: P1 location run +
# P2 Vanilla = 0.57 + 0.87 = 1.44 of one GPU, so this pairing is kept; nothing runs beside ISNet (0.91 with P2), so the
# queue waits while an ISNet run is active. Each step is skipped when its output exists; each training run resumes from
# its own checkpoint (P4) or restarts (P1 location: their code saves the best model only, a run is ~5 h).
#   nohup scripts/replication/short_queue.sh &
cd /root/Where_shortcut_sits
Q=scripts/replication; D=/workspace/data; LOC=$D/replication/viviano/location
isnet_running(){ ps -eo args | grep -q "[p]3_train.py --name isnet"; }
wait_turn(){ while isnet_running || [ -e $D/replication/ISNET_CHAIN_ACTIVE ]; do sleep 300; done; }
step(){ local done_file=$1 need=$2; shift 2
  [ -e "$done_file" ] && return 0
  wait_turn
  echo "$(date +%F_%T) start: $*"
  WTSS_DATA=$D $Q/gpu_wait.sh $need "$@"
  echo "$(date +%F_%T) end:   $*"; }
for pos in inside outside; do
  mkdir -p $LOC/$pos
  # the inside run was started by hand on 2026-11-17 13:13; wait for it instead of starting a second copy
  while ps -eo args | grep -q "[p]1_location.py train --position $pos"; do sleep 300; done
  step $LOC/$pos/results/xray-location_$pos/stats_1234.pkl 6200 \
    bash -c "/workspace/envs/viviano/bin/python -u $Q/p1_location.py train --position $pos >> $LOC/$pos/train.log 2>&1"
  step $LOC/$pos/scores.csv.gz 3000 \
    bash -c "/workspace/envs/viviano/bin/python -u $Q/p1_location.py score --position $pos >> $LOC/$pos/score.log 2>&1"
done
for k in 1 2 3; do
  step $D/replication/saab/runs/erm_split$k/preds_test.csv 8500 \
    bash -c "/workspace/envs/saab/bin/python -u $Q/p4_train.py --method erm --split $k >> $D/replication/saab/erm_split$k.log 2>&1"
done
for k in 1 2 3; do
  step $D/replication/saab/runs/seg_split$k/preds_test.csv 8000 \
    bash -c "/workspace/envs/saab/bin/python -u $Q/p4_train.py --method seg --split $k --grad_ckpt >> $D/replication/saab/seg_split$k.log 2>&1"
done
echo "$(date +%F_%T) short queue done"
