#!/bin/bash
# Phase 5 (PREREGISTRATION_PHASE5.md, approved 2026-11-20): ERM ResNet-50d bases -> arm d -> RR-ClArC grid (3 lambda x 3
# layers) -> ISNet (ImageNet init, alone), Trap A and Trap B x folds 0-2. Each run skips if its preds exist (fit() resumes
# from checkpoint.pt). Each finished run's preds.csv.gz / log.csv are staged for the pusher. Selection on val_clean happens
# in the analysis; test predictions of unselected RR-ClArC configurations are never read.
#   WTSS_DATA=/workspace/data nohup scripts/replication/m2_p5.sh > /workspace/logs/m2_p5.log 2>&1 &
export WTSS_DATA=${WTSS_DATA:-/workspace/data}; D=$WTSS_DATA; Q=/root/Where_shortcut_sits/scripts/replication; PY=/workspace/venv-wtss/bin/python
cd /root/Where_shortcut_sits
run(){ local trap=$1 fold=$2 name=$3; shift 3; local o=$D/phase5/${trap}_f$fold/$name
  [ -s $o/preds.csv.gz ] && return 0
  echo "$(date -u +%F_%T) start $trap f$fold $name"
  # ISNet (13.8 GB) only on an empty GPU; the small networks (<= 5.4 GB) beside at most one other job (P6 prerequisites)
  if [ "$name" = isnet ]; then mo=0; need=15000; else mo=1; need=7000; fi
  MAX_OTHER_JOBS=$mo $Q/gpu_wait.sh $need $PY scripts/phase5/p5_run.py --trap $trap --fold $fold "$@" || { echo "FAILED P5 $trap f$fold $name"; exit 1; }
  local s=$D/results_push/results/replication/phase5/${trap}_f$fold/$name; mkdir -p $s; cp $o/preds.csv.gz $o/log.csv $s/
  echo "$(date -u +%F_%T) done $trap f$fold $name"; }
for trap in trapA trapB; do for f in 0 1 2; do run $trap $f erm_r50d --net erm_r50d; done; done
for trap in trapA trapB; do for f in 0 1 2; do run $trap $f armd --net armd; done; done
for trap in trapA trapB; do for f in 0 1 2; do for lam in 5e8 5e9 5e10; do for layer in identity_1 identity_2 last_conv; do
  n=$($PY -c "print(f'rrclarc_lam{float(\"$lam\"):g}_$layer')"); run $trap $f $n --net rrclarc --lam $lam --layer $layer
done; done; done; done
for trap in trapA trapB; do for f in 0 1 2; do run $trap $f isnet --net isnet --isnet_pretrained; done; done
echo "$(date -u +%F_%T) P5 runs done"
