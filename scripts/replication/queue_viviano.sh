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
p2_running(){ ps -eo args | grep -q "[m]odel_training.start_training"; }
gpu_jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
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
  # P3 interleave (author, 2026-11-17): once the gate is open, the next free shared slot goes to P3 ISNet on P3's original
  # corner benchmark (seed 0), so it is trained when BIMCV completes; P1's other seeds follow
  P3MARK=$RUNS/../P3_ISNET_STARTED
  if [ "$s" != 1234 ] && [ -f $RUNS/../GATE_OK ] && [ ! -f $P3MARK ]; then
    while [ $(gpu_jobs) -ge 2 ] || [ $(free_mib) -lt 11500 ]; do sleep 60; done
    echo "$(date +%F_%T) start P3 ISNet corner/all seed 0"; touch $P3MARK
    nohup $Q/run_p3.sh isnet_corner_all_seed0 --model isnet --position corner --presence all --seed 0 --batch 16 >/dev/null 2>&1 &
    sleep 300
  fi
  # P2 priority (author, 2026-11-17): after the gate, while P2 trains, P2 has at most one other job beside it
  cap=$N; [ "$s" != 1234 ] && p2_running && cap=1
  while [ $(running) -ge $cap ] || [ $(gpu_jobs) -ge $((cap + 1)) ] || [ $(free_mib) -lt $need ]; do sleep 60; done
  echo "$(date +%F_%T) start $t $m $s"; nohup $Q/run_viviano.sh $t $m $s >/dev/null 2>&1 &
  sleep 120
done
echo "$(date +%F_%T) queue: all runs started"
