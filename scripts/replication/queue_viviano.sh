#!/bin/bash
# P1 runs kept under amendment A12 (lean scope), one at a time when the GPU allows it.
# Seed 1234 runs are done (gate passed on the kept runs, A8/A12). Kept: cardiomegaly Standard 3232, 3221 (Masked is scored
# from them, A5); RSNA ActDiff 3232, 3221. A run starts only when at most one other GPU job runs and the GPU has enough free
# memory for its method (scheduling rules 2-4). A run is skipped if its directory exists (finished, running, or started).
Q=/root/Where_shortcut_sits/scripts/replication
RUNS=${WTSS_DATA:-/workspace/data}/replication/viviano/runs
gpu_jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do [ -d /proc/$p ] && echo $p; done | wc -l; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for job in "xray resnet 3232" "rsna resnet_actdiff 3232" "xray resnet 3221" "rsna resnet_actdiff 3221"; do
  set -- $job; t=$1; m=$2; s=$3
  [ -d $RUNS/${t}_${m}_${s} ] && continue
  need=$([ "$m" = resnet_actdiff ] && echo 10000 || echo 6200)   # MiB measured per method (+ margin)
  until [ $(gpu_jobs) -le 1 ] && [ $(free_mib) -ge $need ]; do sleep 120; done
  echo "$(date +%F_%T) start $t $m $s"; nohup $Q/run_viviano.sh $t $m $s >/dev/null 2>&1 &
  sleep 300
done
echo "$(date +%F_%T) P1 queue: all kept runs started"
