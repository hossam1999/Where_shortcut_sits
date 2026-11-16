#!/bin/bash
# Keep up to N P1 runs going at once (default 2) until every registered run exists. A run starts only when the GPU has\n# enough free memory for its method, so a P2/P3 job can share the GPU.
# Order: seed-major, so complete method comparisons accumulate early.
# A run is skipped if its directory exists (finished, running, or started elsewhere).
N=${N:-2}; Q=/root/Where_shortcut_sits/scripts/replication
RUNS=${WTSS_DATA:-/workspace/data}/replication/viviano/runs
SEEDS="1234 3232 3221"   # amendment A3: first three of P1's seed list
running(){  # training processes whose parent is not itself a training process (DataLoader workers excluded)
  ps -eo pid,ppid,args | awk '/activmask\/main.py train/ && !/awk/ {p[$1]=$2} END {n=0; for (i in p) if (!(p[i] in p)) n++; print n}'
}
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
jobs_list(){
  for s in $SEEDS; do for t in rsna xray; do for m in resnet resnet_maskall resnet_actdiff; do echo "$t $m $s"; done; done; done   # A4: best method + baseline only
}
jobs_list | while read t m s; do
  [ -d $RUNS/${t}_${m}_${s} ] && continue
  [ "$t $m" = "xray resnet_maskall" ] && continue   # A5: scored from the Standard model (identical training config)
  # gate (author, 2026-11-16): seeds other than 1234 wait until p1_gate.py has passed (GATE_OK); GATE_FAIL stops the queue
  while [ "$s" != 1234 ] && [ ! -f $RUNS/../GATE_OK ]; do
    [ -f $RUNS/../GATE_FAIL ] && { echo "$(date +%F_%T) gate failed: P1 paused for the author"; exit 0; }
    WTSS_DATA=${WTSS_DATA:-/workspace/data} /workspace/venv-wtss/bin/python $Q/p1_gate.py > $RUNS/../gate.txt 2>&1; sleep 300
  done
  need=$([ "$m" = resnet_actdiff ] && echo 10000 || echo 6200)   # MiB measured per method (+ margin)
  while [ $(running) -ge $N ] || [ $(free_mib) -lt $need ]; do sleep 60; done
  echo "$(date +%F_%T) start $t $m $s"; nohup $Q/run_viviano.sh $t $m $s >/dev/null 2>&1 &
  sleep 120
done
echo "$(date +%F_%T) queue: all runs started"
