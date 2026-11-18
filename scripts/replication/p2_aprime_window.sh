#!/bin/bash
# Short ISNet pause for arm a' (A19; author, 2026-11-18): stop the ISNet chain at its next epoch checkpoint, extract the
# selected A18 RR-ClArC network's features alone (unmasked, as arm a), resume ISNet (also on failure), then the CPU head.
#   nohup scripts/replication/p2_aprime_window.sh > /workspace/logs/p2_aprime_window.log 2>&1 &
set -uo pipefail
ROOT=/root/Where_shortcut_sits; cd $ROOT
Q=$ROOT/scripts/replication; D=/workspace/data; R=$D/replication/pahde
PYP=/workspace/envs/pahde/bin/python; PYW=/workspace/venv-wtss/bin/python
export WTSS_DATA=$D
log(){ echo "$(date +%F_%T) $*"; }
resume_isnet(){ pgrep -f "[p]3_chain.sh" >/dev/null || { nohup $Q/p3_chain.sh >> /workspace/logs/p3_chain.log 2>&1 & log "ISNet chain restarted"; }; }
CH=$($PYW -c "import pandas as pd; d=pd.read_csv('results/replication/pahde/rrclarc_selection_a18.csv'); print(d[d.chosen].model.iloc[0])")
log "arm a' network: $CH"
if [ ! -f $R/scores_a18/aprime_features_unmasked.npz ]; then
  for p in $(pgrep -f "[p]3_chain.sh"); do kill $p && log "stopped p3_chain controller $p"; done
  RUN=$(ls -td $D/replication/isnet/runs/isnet_* | head -1)
  ISN=$(pgrep -f "[p]3_train.py --name isnet" | while read p; do ps -o args= -p $(ps -o ppid= -p $p) | grep -qv p3_train && echo $p; done | head -1)
  if [ -n "$ISN" ]; then
    m0=$(stat -c %Y $RUN/checkpoint.pt); until [ $(stat -c %Y $RUN/checkpoint.pt) -ne $m0 ]; do sleep 5; done; sleep 3
    kill $ISN && log "ISNet $ISN stopped after epoch $(tail -1 $RUN/train_log.csv | cut -d, -f1) checkpoint ($RUN)"
    sleep 20
    for w in $(pgrep -f "[p]3_train.py --name isnet"); do [ "$(ps -o ppid= -p $w | tr -d ' ')" = 1 ] && kill -9 $w; done
  fi
  rm -f $D/replication/ISNET_CHAIN_ACTIVE
  MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 4000 $PYP -u $Q/p2_aprime_features.py --config $R/mitigation_a18/$CH.yaml \
    --ckpt $R/checkpoints_corrected_a18/$CH/last.ckpt --input unmasked > $R/aprime_features.log 2>&1 || log "FAILED: arm a' features"
fi
resume_isnet
[ -f $R/scores_a18/aprime_features_unmasked.npz ] || exit 1
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$ROOT/src nice -n 5 $PYW -u $Q/p2_arms.py --arm aprime > $R/arm_aprime.log 2>&1 || { log "FAILED: arm a' head"; exit 1; }
log "arm a' done"
