#!/bin/bash
# P8 step 1 (A32) with lanes: each lane takes the next unclaimed job from ERM1, DISC1, ERM2, DISC2, ... ERM5, DISC5 (DISC k only
# after ERM k is complete), skipping finished, running or claimed jobs (claim = mkdir lock). Same commands as m2_p8_queue.sh.
#   LANE=a nohup scripts/replication/m2_p8_lanes.sh &   (a second lane: LANE=b, started by m2_pair_p8p8.sh; LANE=c by m2_pair_p8x3.sh)
# GPU concurrency: before each job, MAX_OTHER_JOBS for gpu_wait is read from $WTSS_DATA/disc/max_other_jobs (default 1; the
# pairing scripts set it). Adopting a running job (lane shell replaced): WAIT_PID=<pid> WAIT_JOB="disc 1" waits for that
# process, then logs FAILED and exits 1 if the job's output is incomplete (the watchdog retries it once), else continues.
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; E=/workspace/third_party/disc; PY=/workspace/envs/disc/bin/python
R=$D/disc; L=$D/disc/output; K=$D/disc/locks; mkdir -p $K
erm_dir(){ echo "$L/ISIC/ERM/reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-trapset_id=$1"; }
disc_dir(){ ls -d $L/ISIC/DISC/*-trapset_id=$1 2>/dev/null | head -1; }
complete(){ local d=$1; [ -n "$d" ] && [ -s "$d/test.csv" ] && [ $(wc -l < "$d/test.csv") -ge 101 ]; }
# A37: one data root per trap set. DISC's ISICDataset caches the transformed images at <root_dir>/features/feature_mat.npy
# without the trap-set id, so a shared root makes sets 2-5 reuse set 1's image order with their own labels.
rootk(){ local r=$R/isic_set$1; if [ ! -d $r ]; then mkdir -p $r; ln -s $R/isic/ISIC2018_Task1-2_Training_Input $r/ISIC2018_Task1-2_Training_Input
  ln -s $R/isic/trap-sets $r/trap-sets; [ $1 = 1 ] && [ -s $R/isic/features/feature_mat.npy ] && ln -s $R/isic/features $r/features; fi; echo $r/; }
maxo(){ local n; n=$(cat $D/disc/max_other_jobs 2>/dev/null); echo ${n:-1}; }
if [ -n "$WAIT_PID" ]; then
  set -- $WAIT_JOB; echo "$(date -u +%F_%T) lane ${LANE:-a}: adopting running $1 trap set $2 (pid $WAIT_PID)"
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
  if [ $1 = erm ]; then wd=$(erm_dir $2); else wd=$(disc_dir $2); fi
  complete "$wd" || { echo "FAILED $(echo $1 | tr a-z A-Z) $2"; rmdir $K/${1}_$2; exit 1; }
  set --
fi
while :; do
  job=""; for k in ${SPLITS:-1 2 3 4 5}; do for m in erm disc; do
    if [ $m = erm ]; then complete "$(erm_dir $k)" && continue; else complete "$(disc_dir $k)" && continue; complete "$(erm_dir $k)" || continue; fi
    # already running (any queue)? the ERM command line ENDS with "--seed k", so match a space or end of line after k
    [ $m = erm ] && ps -eo args | grep "^$PY run_expt.py" | grep -v -- "--disc" | grep -Eq -- "--seed $k( |$)" && continue
    [ $m = disc ] && ps -eo args | grep "^$PY run_expt.py" | grep -- "--disc" | grep -Eq -- "--seed $k( |$)" && continue
    mkdir $K/${m}_$k 2>/dev/null || continue
    job="$m $k"; break 2
  done; done
  [ -z "$job" ] && { pending=0; for k in ${SPLITS:-1 2 3 4 5}; do complete "$(disc_dir $k)" || pending=1; done; [ $pending = 0 ] && break; sleep 60; continue; }
  set -- $job; m=$1; k=$2; echo "$(date -u +%F_%T) lane ${LANE:-a}: $m trap set $k"
  # A37 (author): row-by-row check that trap set k's cache (if it exists yet) belongs to trap set k, before every run
  $PY $Q/p8_cache_check.py $k --pre --root $(rootk $k) || { echo "FAILED CACHE CHECK $m $k (not started)"; rmdir $K/${m}_$k; exit 1; }
  if [ $m = erm ]; then
    (cd $E && MAX_OTHER_JOBS=$(maxo) $Q/gpu_wait.sh 4000 $PY run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 \
      --weight_decay 1e-5 --model resnet50 --n_epochs 100 --root_dir $(rootk $k) --log_dir $L/ --save_best --save_last --seed $k \
      > $D/disc/erm_$k.log 2>&1) || { echo "FAILED ERM $k"; rmdir $K/erm_$k; exit 1; }
    $PY $Q/p8_cache_check.py $k --root $(rootk $k) || { echo "FAILED CACHE CHECK erm $k (cache built by the run)"; exit 1; }
  else
    (cd $E && MAX_OTHER_JOBS=$(maxo) $Q/gpu_wait.sh 4000 $PY run_expt.py -s confounder -d ISIC -t label -c hair --lr 5e-4 --batch_size 16 \
      --weight_decay 1e-5 --model resnet50 --n_epochs 100 --root_dir $(rootk $k) --log_dir $L/ --erm_path $(erm_dir $k)/best_model.pth \
      --concept_img_folder $R/synthetic_concepts --concept_categories color-texture --n_clusters 3 --save_best --save_last --seed $k --disc \
      > $D/disc/disc_$k.log 2>&1) || { echo "FAILED DISC $k"; rmdir $K/disc_$k; exit 1; }
  fi
done
echo "$(date -u +%F_%T) lane ${LANE:-a}: P8 step-1 runs done" | tee -a /workspace/logs/m2_p8_queue.log
