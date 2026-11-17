#!/bin/bash
# P3 ISNet chain under A12 (critical path, one run after another; each run resumes from its checkpoint if restarted):
#   1. original corner benchmark (P3 design: every image marked)   2. marks inside the lungs, P3 design   3. inside, 90/10
# ISNet (~10.4 GB) only beside one small job: a run starts when >= 11,500 MiB are free and at most one other GPU job runs.
Q=/root/Where_shortcut_sits/scripts/replication; D=${WTSS_DATA:-/workspace/data}/replication/isnet/runs
gpu_jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do [ -d /proc/$p ] && echo $p; done | wc -l; }
run(){ local name=$1; shift
  [ -f $D/$name/best.pt ] && [ "$(tail -1 $D/$name/train_log.csv 2>/dev/null | cut -d, -f1)" = 47 ] && return 0
  until [ $(gpu_jobs) -le 1 ]; do sleep 60; done
  $Q/gpu_wait.sh 11500 $Q/run_p3.sh $name --model isnet --seed 0 --batch 16 "$@"
  echo "$(date +%F_%T) finished $name"; }
run isnet_corner_all_seed0 --position corner --presence all
run isnet_inside_all_seed0 --position inside --presence all
run isnet_inside_9010_seed0 --position inside --presence 9010
echo "$(date +%F_%T) P3 ISNet chain done"
