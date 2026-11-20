#!/bin/bash
# Queue after the P8 runs (author, 2026-11-19): P8 gate -> our P8 arms (A32/A35/A37, both selection rules) -> P6 A34 (A36).
# Waits for "P8 step-1 runs done"; each step stops the chain on failure ("FAILED P8POST <step>", the watchdog alerts).
# Results go to $WTSS_DATA/results_push/... and are pushed by m2_push_results.sh.
#   WTSS_DATA=/workspace/data nohup scripts/replication/m2_p8_post.sh > /workspace/logs/m2_p8_post.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; PYW=/workspace/venv-wtss/bin/python; PYD=/workspace/envs/disc/bin/python
until grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log && ! ps -eo args | grep -q "^$PYD run_expt.py"; do sleep 120; done
step(){ local n=$1; shift; echo "$(date -u +%F_%T) start $n"; "$@" || { echo "$(date -u +%F_%T) FAILED P8POST $n"; exit 1; }; echo "$(date -u +%F_%T) done $n"; }
cd /root/Where_shortcut_sits
[ -s $D/results_push/results/replication/disc/gate.json ] || step gate $PYW $Q/p8_gate.py > /workspace/logs/m2_p8_gate.out   # the gate's own text says PASSED/FAILED; kept out of this log (watchdog)
step features_a env MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 8000 $PYD $Q/p8_arms.py --stage features_a
step features_bc env MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 8000 $PYW $Q/p8_arms.py --stage features_bc
step armd env MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 8000 $PYW $Q/p8_arms.py --stage armd
step heads $PYW $Q/p8_arms.py --stage heads
step compare $PYW $Q/p8_arms.py --stage compare
echo "$(date -u +%F_%T) P8 arms done; starting P6 A34 (A36 mixup)"
grep -q "A36 (paper Eq. 3)" /workspace/third_party/epvt/domainbed/algorithms.py || { echo "FAILED P8POST A36 patch missing"; exit 1; }
nohup $Q/m2_p6_5seeds.sh > /workspace/logs/m2_p6_5seeds.log 2>&1 &
