#!/bin/bash
# Author 2026-11-19: when the P6 A33 rerun has finished, pair two P8 trap sets: lane b starts beside lane a (concurrency file
# $WTSS_DATA/disc/max_other_jobs = 1, i.e. two GPU jobs). Measured by p8_measure.py on the two concurrent jobs (common window,
# s/epoch vs alone: ERM 9.6 measured, DISC from pair_p6p8.json; CPU busy % recorded). Kept only if total > 1.0; otherwise lane b
# and its job are stopped (process tree; folder set aside) and max_other_jobs = 0, so lane a runs one job at a time.
# Result: /workspace/logs/pair_p8p8.json. (v2 11:10: lane-tree stop and concurrency file; v1 left lane a's job orphaned.)
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; . $Q/p8_lane_lib.sh
until { [ -d $D/epvt/runs/preA33_epvt_bias1_1 ] && [ -s $D/epvt/runs/erm_bias1_1/erm__test_bias_1_1.json ] && [ -s $D/epvt/runs/epvt_bias1_1/epvt__test_bias_1_1.json ]; } \
      || grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log; do sleep 120; done
grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log && { echo "P8 already done; nothing to pair"; exit 0; }
echo "$(date -u +%T) P6 A33 done; starting lane b beside lane a"
echo 1 > $D/disc/max_other_jobs
(cd /root/Where_shortcut_sits && LANE=b WTSS_DATA=$D nohup $Q/m2_p8_lanes.sh >> /workspace/logs/m2_p8_queue.log 2>&1 &)
until python3 $Q/p8_measure.py /workspace/logs/pair_p8p8.json 2 1200 && python3 -c "import json,sys; sys.exit(0 if json.load(open('/workspace/logs/pair_p8p8.json'))['valid'] else 1)"; do
  echo "$(date -u +%T) measurement window too short (a job ended); measuring again"; done
if python3 -c "import json,sys; sys.exit(0 if json.load(open('/workspace/logs/pair_p8p8.json'))['total'] > 1.0 else 1)"; then
  echo "$(date -u +%T) P8 two-lane pairing KEPT"; exit 0; fi
echo "$(date -u +%T) P8 two-lane pairing NOT kept: stopping lane b and its job; one job at a time"
echo 0 > $D/disc/max_other_jobs; stop_lane b
