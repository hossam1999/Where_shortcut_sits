#!/bin/bash
# Arm d on P4 (author, 2026-11-18): alone on the GPU at the registered batch 64 (512 px does not fit beside ISNet), in a
# short ISNet pause. Holds the GPU launcher's lock for the whole run so that nothing else starts beside it; waits for the
# P3 test cache and the COVIDIsNet check first (priority), then until only ISNet is left; stops ISNet right after its
# next epoch checkpoint; runs P4 train + features; releases the lock; restarts the ISNet chain (also on failure); head on CPU.
#   nohup scripts/replication/armd_p4_window.sh > /workspace/logs/armd_p4_window.log 2>&1 &
set -uo pipefail
ROOT=/root/Where_shortcut_sits; cd $ROOT
Q=$ROOT/scripts/replication; D=/workspace/data; A=$D/replication/armd/p4_split1; PY=/workspace/venv-wtss/bin/python
export WTSS_DATA=$D PYTHONPATH=$ROOT/src
log(){ echo "$(date +%F_%T) $*"; }
jobs_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do [ -d /proc/$p ] && echo $p; done | wc -l; }
resume_isnet(){ pgrep -f "[p]3_chain.sh" >/dev/null || { nohup $Q/p3_chain.sh >> /workspace/logs/p3_chain.log 2>&1 9>&- & log "ISNet chain restarted"; }; }
mkdir -p $A
until [ -f $D/replication/isnet/cache_test/ids.txt ] && [ -f $ROOT/results/replication/isnet/covidisnet_check.md ]; do sleep 60; done
log "P3 test cache and COVIDIsNet check done"
exec 9>/tmp/wtss_gpu_adhoc2.lock; flock 9; log "launcher lock held"
until [ "$(jobs_on_gpu)" -le 1 ]; do sleep 20; done; log "only ISNet left on the GPU"
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
until [ "$(jobs_on_gpu)" -eq 0 ]; do sleep 5; done
ok=1
[ -f $A/TRAIN_DONE ] || $PY -u $Q/armd_train.py --setting p4_split1 --stage train >> $A/train.log 2>&1 9>&- || { ok=0; log "FAILED: P4 train"; }
[ $ok = 1 ] && { [ -f $A/features.npz ] || $PY -u $Q/armd_train.py --setting p4_split1 --stage features >> $A/features.log 2>&1 9>&- || { ok=0; log "FAILED: P4 features"; }; }
flock -u 9; exec 9>&-; log "launcher lock released"
resume_isnet
[ $ok = 1 ] || exit 1
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= nice -n 5 $PY -u $Q/armd_train.py --setting p4_split1 --stage heads > $A/heads.log 2>&1 && log "arm d P4 done" || log "FAILED: P4 heads"
