#!/bin/bash
# A30 re-run (author, 2026-11-20): every affected P3 arm result once with the fix, + A24 for these arms. GPU via the launcher.
cd /root/Where_shortcut_sits; export WTSS_DATA=/workspace/data PYTHONPATH=src
PY=/workspace/venv-wtss/bin/python; Q=scripts/replication; I=$WTSS_DATA/replication/isnet; A=$WTSS_DATA/replication/armd
log(){ echo "$(date +%F_%T) $*"; }
cpu(){ OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= nice -n 5 "$@"; }
# 1. arm c, corner: add the A24 validation features, refit (deterministic) -> A24 rows
P3_SETTING=corner $Q/gpu_wait.sh 3000 $PY -u $Q/p3_arms_bc.py --stage features >> $I/arms_bc_corner/run.log 2>&1 && P3_SETTING=corner cpu $PY -u $Q/p3_arms_bc.py --stage heads >> $I/arms_bc_corner/run.log 2>&1 && log "arm c corner A24 done" || log "FAILED arm c corner A24"
# 2. arms b, c on the 90/10 inside variant
[ -d $I/arms_bc_invalid_A30 ] || mv $I/arms_bc $I/arms_bc_invalid_A30; mkdir -p $I/arms_bc
$Q/gpu_wait.sh 3000 $PY -u $Q/p3_arms_bc.py --stage features >> $I/arms_bc/run.log 2>&1 && cpu $PY -u $Q/p3_arms_bc.py --stage heads >> $I/arms_bc/run.log 2>&1 && log "arms b, c inside 90/10 done" || log "FAILED arms b, c inside"
# 3. arm d on the 90/10 inside variant: encoder retrained
[ -d $A/p3_9010_inside_invalid_A30 ] || mv $A/p3_9010_inside $A/p3_9010_inside_invalid_A30; mkdir -p $A/p3_9010_inside
$Q/gpu_wait.sh 3000 $PY -u $Q/armd_train.py --setting p3_9010_inside --stage train >> $A/p3_9010_inside/train.log 2>&1 && \
$Q/gpu_wait.sh 3000 $PY -u $Q/armd_train.py --setting p3_9010_inside --stage features >> $A/p3_9010_inside/features.log 2>&1 && \
cpu $PY -u $Q/armd_train.py --setting p3_9010_inside --stage heads > $A/p3_9010_inside/heads.log 2>&1 && log "arm d inside 90/10 done" || log "FAILED arm d"
log "A30 re-run done"
