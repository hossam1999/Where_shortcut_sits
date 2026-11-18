#!/bin/bash
# Machine 2 (author, 2026-11-18): measure P6 bias-1 ERM paired with P4 A31 split 3; keep the pairing only if the total throughput
# exceeds 1.0 of one GPU (rule 2). Split 3 alone = 1.14 min/epoch (machine-2 Seg-ERM alone, epochs 0-1, 2026-11-18). P6 alone is
# measured by pausing split 3 (SIGSTOP / SIGCONT, by PID incl. its DataLoader workers) for three P6 epochs. Epoch times include
# validation (as each job runs). Result: /workspace/logs/pair_measure.json.
D=${WTSS_DATA:-/workspace/data}; L3=$D/replication/saab/seg_pil_split3.log; P6L=$D/epvt/runs/erm_bias1_1/train.log
Q=/root/Where_shortcut_sits/scripts/replication; S3_ALONE=1.14
ep6(){ local n; n=$(grep -c "^Epoch [0-9]" $P6L 2>/dev/null); echo ${n:-0}; }
ep3(){ local n; n=$(grep -c "^{'epoch'" $L3 2>/dev/null); echo ${n:-0}; }
until [ "$(ep6)" -ge 1 ]; do sleep 20; done
n3=$(ep3); echo "$(date -u +%T) P6 training; split 3 at $n3 epochs"
until [ "$(ep3)" -ge $((n3 + 3)) ] && [ "$(ep6)" -ge 4 ]; do sleep 20; done      # 2 split-3 epochs entirely paired (skip the first)
S3P=$(grep "^{'epoch'" $L3 | tail -2 | sed -E "s/.*'minutes': ([0-9.]+).*/\1/" | awk '{s+=$1} END {print s/NR}')
P6P=$(grep "^Epoch [0-9]" $P6L | tail -3 | sed -E 's/.*Training time: ([0-9:]+).*/\1/' | awk -F: '{s+=$1*3600+$2*60+$3} END {print s/NR}')
MAIN=$(ps -eo pid,args | awk '$2 ~ /envs\/saab\/bin\/python/ && /--split 3 / && /seg_pil_split3/ {print $1}' | head -1)
KIDS=$(pgrep -P $MAIN | tr '\n' ' '); echo "$(date -u +%T) pause split 3: pid $MAIN workers $KIDS"
n6=$(ep6); kill -STOP $MAIN $KIDS
timeout 1200 bash -c "until [ \$(grep -c '^Epoch [0-9]' $P6L) -ge $((n6 + 4)) ]; do sleep 10; done"
kill -CONT $MAIN $KIDS; echo "$(date -u +%T) resumed split 3"
P6A=$(grep "^Epoch [0-9]" $P6L | tail -3 | sed -E 's/.*Training time: ([0-9:]+).*/\1/' | awk -F: '{s+=$1*3600+$2*60+$3} END {print s/NR}')
python3 - "$S3P" "$S3_ALONE" "$P6P" "$P6A" <<'PY'
import json, sys
s3p, s3a, p6p, p6a = map(float, sys.argv[1:])
r3, r6 = s3a / s3p, p6a / p6p
r = {"split3_min_per_epoch_paired": s3p, "split3_min_per_epoch_alone": s3a, "split3_ratio": r3,
     "p6_sec_per_epoch_paired": p6p, "p6_sec_per_epoch_alone": p6a, "p6_ratio": r6, "total": r3 + r6, "keep_pairing": r3 + r6 > 1.0}
json.dump(r, open("/workspace/logs/pair_measure.json", "w"), indent=1); print(json.dumps(r))
PY
if python3 -c "import json,sys; sys.exit(0 if json.load(open('/workspace/logs/pair_measure.json'))['keep_pairing'] else 1)"; then
  echo "$(date -u +%T) pairing KEPT"
else
  echo "$(date -u +%T) pairing NOT kept: stopping P6 by PID and re-queueing it after split 3"
  for p in $(ps -eo pid,args | awk '/m2_p6_queue.sh/ && !/awk/ {print $1}'); do kill $p; done
  T=$(ps -eo pid,args | awk '$2 ~ /envs\/epvt\/bin\/python/ && /train_(erm|epvt)/ {print $1}')
  for p in $T; do kill $(pgrep -P $p) $p 2>/dev/null; done
  sleep 10; mv $D/epvt/runs/erm_bias1_1 $D/epvt/runs/paired_stopped_erm_bias1_1_$(date -u +%H%M)
  cd /root/Where_shortcut_sits && P6_START=after nohup $Q/m2_p6_queue.sh > /workspace/logs/m2_p6_queue_after.log 2>&1 &
fi
