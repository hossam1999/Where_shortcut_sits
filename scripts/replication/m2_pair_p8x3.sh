#!/bin/bash
# Author 2026-11-19: third P8 lane, measured like the two-lane pairing, after it is kept. Lane c starts with the concurrency
# file at 2 (three GPU jobs). Kept only if (i) total throughput > 1.0 of one GPU, (ii) whole-machine CPU busy < 85 % of the
# 28 cores over the window (DISC's concept step is CPU-bound), and (iii) total > the two-lane total (pair_p8p8.json), so that a
# third lane is not kept when it lowers throughput. Otherwise lane c and its job are stopped (process tree; folder set
# aside) and the file goes back to 1 (two lanes). Result: /workspace/logs/pair_p8x3.json.
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; . $Q/p8_lane_lib.sh
J2=/workspace/logs/pair_p8p8.json; J3=/workspace/logs/pair_p8x3.json
until [ -s $J2 ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$J2')).get('valid') else 1)"; do
  grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log && { echo "P8 done; no third lane"; exit 0; }; sleep 60; done
python3 -c "import json,sys; sys.exit(0 if json.load(open('$J2'))['total'] > 1.0 else 1)" || { echo "two lanes not kept; no third lane"; exit 0; }
sleep 30   # let the two-lane script finish its bookkeeping
echo "$(date -u +%T) starting lane c (three GPU jobs)"
echo 2 > $D/disc/max_other_jobs
(cd /root/Where_shortcut_sits && LANE=c WTSS_DATA=$D nohup $Q/m2_p8_lanes.sh >> /workspace/logs/m2_p8_queue.log 2>&1 &)
until python3 $Q/p8_measure.py $J3 3 1500 && python3 -c "import json,sys; sys.exit(0 if json.load(open('$J3'))['valid'] else 1)"; do
  echo "$(date -u +%T) measurement window too short (a job ended); measuring again"; done
python3 - $J2 $J3 <<'PY'
import json, sys
j2, j3 = (json.load(open(f)) for f in sys.argv[1:])
j3["two_lane_total"] = j2["total"]
j3["keep"] = j3["total"] > 1.0 and j3["cpu_busy_pct"] < 85 and j3["total"] > j2["total"]
json.dump(j3, open(sys.argv[2], "w"), indent=1); print(json.dumps({k: j3[k] for k in ("total", "two_lane_total", "cpu_busy_pct", "keep")}))
PY
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$J3'))['keep'] else 1)"; then echo "$(date -u +%T) third lane KEPT"; exit 0; fi
echo "$(date -u +%T) third lane NOT kept: stopping lane c and its job; back to two lanes"
echo 1 > $D/disc/max_other_jobs; stop_lane c
