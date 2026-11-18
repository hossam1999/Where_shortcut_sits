#!/bin/bash
# Arm d (A20/A21) settings one after another: train (GPU), features (GPU), head (CPU). GPU steps through the launcher
# (at most 2 GPU jobs; beside ISNet as measured on P2: 1.32). Each step is skipped when its output exists; training resumes
# from its checkpoint.   nohup scripts/replication/armd_queue.sh p1_cardio p1_rsna > /workspace/logs/armd_queue.log 2>&1 &
cd /root/Where_shortcut_sits; export WTSS_DATA=/workspace/data PYTHONPATH=src
PY=/workspace/venv-wtss/bin/python; Q=scripts/replication; A=$WTSS_DATA/replication/armd
for s in "$@"; do
  mkdir -p $A/$s
  [ -f $A/$s/TRAIN_DONE ] || $Q/gpu_wait.sh 3000 $PY -u $Q/armd_train.py --setting $s --stage train >> $A/$s/train.log 2>&1 || { echo "$(date +%F_%T) FAILED train $s"; continue; }
  [ -f $A/$s/features.npz ] || $Q/gpu_wait.sh 3000 $PY -u $Q/armd_train.py --setting $s --stage features >> $A/$s/features.log 2>&1 || { echo "$(date +%F_%T) FAILED features $s"; continue; }
  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= nice -n 5 $PY -u $Q/armd_train.py --setting $s --stage heads > $A/$s/heads.log 2>&1 \
    && echo "$(date +%F_%T) done $s" || echo "$(date +%F_%T) FAILED heads $s"
done
