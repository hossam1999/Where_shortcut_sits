#!/bin/bash
# machine 1 (author's order, 2026-11-19): ISNet run 3 -> its A24 evaluation (CPU) -> arm c corner (already running) ->
# exploratory candidate (b) for alpha in {0.1, 1, 10} (GPU via the launcher; validation only).
#   nohup scripts/newmethod/machine1_rest.sh > /workspace/logs/machine1_rest.log 2>&1 &
cd /root/Where_shortcut_sits; export WTSS_DATA=/workspace/data PYTHONPATH=src
R=$WTSS_DATA/replication/isnet/runs/isnet_inside_9010_seed0
log(){ echo "$(date +%F_%T) $*"; }
until [ -f $R/preds_deceiving.csv.gz ] && ! pgrep -f "[p]3_train.py --name isnet_inside_9010" >/dev/null; do sleep 120; done
log "ISNet run 3 scored"
[ -f results/replication/isnet/valid_secondary_isnet_inside_9010_seed0.md ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=32 nice -n 5 \
  /workspace/envs/pahde/bin/python -u scripts/replication/p3_valid_eval.py --name isnet_inside_9010_seed0 --position inside --presence 9010 \
  > $WTSS_DATA/replication/isnet/valid_eval_inside_9010.log 2>&1 && log "run 3 A24 done"
until [ -f results/replication/isnet/p3_corner_arm_c.md ] || grep -q Traceback $WTSS_DATA/replication/isnet/arms_bc_corner/run.log; do sleep 60; done
log "arm c corner done"
for a in 0.1 1 10; do
  O=$WTSS_DATA/newmethod/adv_p1loc_inside_a$a; mkdir -p $O
  [ -f $O/TRAIN_DONE ] || scripts/replication/gpu_wait.sh 3000 /workspace/venv-wtss/bin/python -u scripts/newmethod/adv_train.py --alpha $a --stage train >> $O/train.log 2>&1 || { log "FAILED train $a"; continue; }
  [ -f $O/features_trainval.npz ] || scripts/replication/gpu_wait.sh 3000 /workspace/venv-wtss/bin/python -u scripts/newmethod/adv_train.py --alpha $a --stage features >> $O/train.log 2>&1 || { log "FAILED features $a"; continue; }
  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= nice -n 5 /workspace/venv-wtss/bin/python -u scripts/newmethod/adv_train.py --alpha $a --stage heads >> $O/train.log 2>&1 && log "candidate (b) alpha $a done"
done
log "machine 1 queue done"
