#!/bin/bash
# Machine-2 queue watchdog (author, 2026-11-19, after the 09:25-10:27 idle hour). Every 30 s:
#  1. a queued job exited non-zero (a new "FAILED <job>" line in a queue log): log it; P8 jobs are retried ONCE (incomplete
#     output folder moved aside as failed_auto_<time>_..._x, lane restarted); a second failure of the same job, or a failure in
#     any other queue (P6 runs are registered "once"), is only alerted;
#  2. the GPU is idle (max utilisation < 5 % in every sample) for > 10 min while jobs are queued: log it; if P8 is incomplete
#     and no P8 lane is alive, restart lane a once per job; if a job process is alive but idle (a hang), alert only.
# Each event: one dated line inserted (newest first) under the "Watchdog log" heading of docs/REPLICATION_STATUS.md (committed with the next machine-2 commit) and to
# /workspace/logs/watchdog_alerts.log (watched by the session). Log: /workspace/logs/m2_watchdog.log.
#   WTSS_DATA=/workspace/data nohup scripts/replication/m2_watchdog.sh > /workspace/logs/m2_watchdog.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; REPO=/root/Where_shortcut_sits; Q=$REPO/scripts/replication; LG=${WD_LOGS:-/workspace/logs}
ST=${WD_STATE:-$D/watchdog}; mkdir -p $ST; ALERTS=$LG/watchdog_alerts.log; STATUS=${WD_STATUS:-$REPO/docs/REPLICATION_STATUS.md}
DRY=${WD_DRY:-0}   # 1 = test mode: log actions, never move folders or start lanes
L=$D/disc/output; K=$D/disc/locks
QUEUE_LOGS="$LG/m2_p5.log $LG/m2_p7.log $LG/m2_p8_post.log $LG/m2_push.log $LG/m2_p8_queue.log $LG/m2_p6_a33.log $LG/m2_p6_5seeds.log $LG/m2_pair_p6p8.log $LG/m2_pair_p8p8.log"
QUEUE_SCRIPTS='m2_p5\.sh|m2_p7\.sh|m2_p8_post\.sh|m2_p8_lanes\.sh|m2_pair_p6p8\.sh|m2_pair_p8p8\.sh|m2_p6_queue\.sh|m2_p6_a33\.sh|m2_p6_5seeds\.sh|gpu_wait\.sh'
JOB_PROCS='run_expt\.py|train_(erm|epvt)|p5_run\.py|p4_train\.py'
IDLE_MIN=${IDLE_MIN:-10}

erm_dir(){ echo "$L/ISIC/ERM/reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-trapset_id=$1"; }
disc_dir(){ ls -d $L/ISIC/DISC/*-trapset_id=$1 2>/dev/null | head -1; }
complete(){ local d=$1; [ -n "$d" ] && [ -s "$d/test.csv" ] && [ $(wc -l < "$d/test.csv") -ge 101 ]; }
p8_incomplete(){ for k in 1 2 3 4 5; do complete "$(disc_dir $k)" || return 0; done; return 1; }
alive(){ ps -eo args | awk -v p="$1" '$0 ~ p && !/awk/ && !/m2_watchdog/' | grep -q .; }
now(){ date -u +%F_%T; }
alert(){   # one line to STATUS (working tree) and to the alerts file
  local msg="$(now) UTC watchdog (machine 2): $*"
  echo "$msg" | tee -a $ALERTS
  sed -i "/^### Watchdog log (machine 2/a - ${msg//\\/}" $STATUS   # newest first, under its own heading
}
start_lane(){ [ $DRY = 1 ] && { echo "DRY start_lane"; return; }; (cd $REPO && LANE=a WTSS_DATA=$D nohup $Q/m2_p8_lanes.sh >> $LG/m2_p8_queue.log 2>&1 &); }
move_aside(){   # incomplete P8 output of job m k
  [ $DRY = 1 ] && { echo "DRY move_aside $*"; return; }
  local m=$1 k=$2 d; [ $m = erm ] && d=$(erm_dir $k) || d=$(disc_dir $k)
  [ -n "$d" ] && [ -d "$d" ] && ! complete "$d" && mv "$d" "$(dirname $d)/failed_auto_$(date -u +%H%M)_$(basename $d)_x"
  rmdir $K/${m}_$k 2>/dev/null
}

grep -q "^### Watchdog log (machine 2" $STATUS || printf '\n### Watchdog log (machine 2, automatic; `scripts/replication/m2_watchdog.sh`)\n' >> $STATUS
nfail(){ local n; n=$(grep -c "FAILED" "$1" 2>/dev/null | head -1); echo ${n:-0}; }
for f in $QUEUE_LOGS; do [ -f $ST/seen_$(basename $f) ] || nfail $f > $ST/seen_$(basename $f); done
idle=0
echo "$(now) watchdog started (idle limit ${IDLE_MIN} min)"
while :; do
  # 1. new failures
  for f in $QUEUE_LOGS; do
    [ -f $f ] || continue
    n=$(nfail $f); s=$(head -1 $ST/seen_$(basename $f) 2>/dev/null); s=${s:-0}
    [ "$n" -le "$s" ] && continue
    grep "FAILED" $f | tail -n $((n - s)) | while read -r line; do
      if [ "$(basename $f)" = m2_p8_queue.log ] && [[ $line =~ FAILED\ (ERM|DISC)\ ([1-5]) ]]; then
        m=$(echo ${BASH_REMATCH[1]} | tr A-Z a-z); k=${BASH_REMATCH[2]}
        if [ ! -f $ST/retried_${m}_$k ]; then
          touch $ST/retried_${m}_$k; move_aside $m $k
          alert "P8 $m trap set $k exited non-zero ($(tail -c 300 $D/disc/${m}_$k.log | tr '\n\r' '  ' | tail -c 200)); retried once (lane a restarted)"
          alive m2_p8_lanes || start_lane
        else
          alert "P8 $m trap set $k failed AGAIN after the retry; NOT retried — needs a look ($D/disc/${m}_$k.log)"
        fi
      else
        alert "queue failure in $(basename $f): '$line' — alert only (not retried)"
      fi
    done
    echo $n > $ST/seen_$(basename $f)
  done
  # 2. idle GPU while jobs are queued
  u=0; for i in 1 2 3 4 5; do v=${WD_FAKE_UTIL:-$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)}; [ "$v" -gt "$u" ] && u=$v; sleep 1; done
  queued=0; alive "$QUEUE_SCRIPTS" && queued=1; p8_incomplete && queued=1
  if [ $queued = 1 ] && [ "$u" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
  if [ $idle -ge $((IDLE_MIN * 2)) ]; then
    if alive "$JOB_PROCS"; then
      alert "GPU idle > ${IDLE_MIN} min while a job process is alive (possible hang): $(ps -eo pid,etime,args | awk '/run_expt|train_erm|train_epvt|p5_run|p4_train/ && !/awk/ {print $1, $2}' | tr '\n' ';') — alert only"
    elif p8_incomplete && ! alive m2_p8_lanes; then
      if [ ! -f $ST/idle_restart_$(date -u +%F_%H) ]; then
        touch $ST/idle_restart_$(date -u +%F_%H); alert "GPU idle > ${IDLE_MIN} min, P8 incomplete and no P8 lane alive: lane a restarted"
        start_lane
      else
        alert "GPU idle > ${IDLE_MIN} min again with no P8 lane; already restarted this hour — needs a look"
      fi
    else
      alert "GPU idle > ${IDLE_MIN} min while queue scripts wait ($(ps -eo args | grep -Eo "$QUEUE_SCRIPTS" | sort -u | tr '\n' ' ')) — alert only"
    fi
    idle=0   # re-alert after another full idle period
  fi
  sleep 25
done
