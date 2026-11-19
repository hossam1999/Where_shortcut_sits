#!/bin/bash
# Author 2026-11-19: when the P6 A33 rerun has finished, pair two P8 trap sets. The old sequential P8 queue's bash shells are
# stopped by PID (its running training job continues); lanes a and b take the next jobs. Measured on the first two concurrent
# jobs: s/epoch paired vs alone (ERM alone 9.6 s/epoch, measured; DISC alone from pair_p6p8.json). Kept only if total > 1.0;
# otherwise lane b and the job it overlapped are stopped by PID and P8 continues with one lane.
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; PY=/workspace/envs/disc/bin/python
# the A33 rerun is done only when the pre-A33 runs have been set aside (by m2_pair_p6p8.sh) AND new result files exist
until { [ -d $D/epvt/runs/preA33_epvt_bias1_1 ] && [ -s $D/epvt/runs/erm_bias1_1/erm__test_bias_1_1.json ] && [ -s $D/epvt/runs/epvt_bias1_1/epvt__test_bias_1_1.json ]; } \
      || grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log; do sleep 120; done
grep -q "P8 step-1 runs done" /workspace/logs/m2_p8_queue.log && { echo "P8 already done; nothing to pair"; exit 0; }
echo "$(date -u +%T) P6 A33 done; starting lane b beside lane a"
cd /root/Where_shortcut_sits
LANE=b nohup $Q/m2_p8_lanes.sh >> /workspace/logs/m2_p8_queue.log 2>&1 &
# measure the first two concurrent jobs
jobs_now(){ ps -eo pid,args | awk -v py="$PY" '$2==py && /run_expt.py/ {print $1}'; }
until [ $(jobs_now | wc -l) -ge 2 ]; do sleep 15; done
DA=$(python3 -c "import json;print(json.load(open('/workspace/logs/pair_p6p8.json'))['disc_s_per_epoch_alone'])" 2>/dev/null || echo "")
csvof(){ local pid=$1 a s k; a=$(ps -o args= -p $pid); k=$(echo "$a" | sed -E 's/.*--seed ([0-9]+).*/\1/');
  if echo "$a" | grep -q -- "--disc"; then ls -d $D/disc/output/ISIC/DISC/*trapset_id=$k-*/test.csv; else echo $D/disc/output/ISIC/ERM/*trapset_id=$k/test.csv; fi; }
P=($(jobs_now)); F1=$(csvof ${P[0]}); F2=$(csvof ${P[1]})
until [ -s "$F1" ] && [ -s "$F2" ] && [ $(wc -l < $F1) -ge 3 ] && [ $(wc -l < $F2) -ge 3 ]; do sleep 10; F1=$(csvof ${P[0]}); F2=$(csvof ${P[1]}); done
a0=$(wc -l < $F1); b0=$(wc -l < $F2); t0=$(date +%s)
until [ $(wc -l < $F1) -ge $((a0 + 6)) ] && [ $(wc -l < $F2) -ge $((b0 + 6)) ]; do sleep 5; done
t1=$(date +%s); a1=$(wc -l < $F1); b1=$(wc -l < $F2)
python3 - "$F1" "$F2" "$a0" "$a1" "$b0" "$b1" "$t0" "$t1" "$DA" <<'PY'
import json, sys
f1, f2 = sys.argv[1:3]; a0, a1, b0, b1, t0, t1 = map(float, sys.argv[3:9]); da = sys.argv[9]
alone = lambda f: (float(da) if "/DISC/" in f else 9.6)
p1, p2 = (t1 - t0) / (a1 - a0), (t1 - t0) / (b1 - b0)
r = {"job1": f1, "job2": f2, "job1_s_per_epoch_paired": p1, "job2_s_per_epoch_paired": p2,
     "job1_alone": alone(f1), "job2_alone": alone(f2)}
r["total"] = alone(f1) / p1 + alone(f2) / p2; r["keep_pairing"] = r["total"] > 1.0
json.dump(r, open("/workspace/logs/pair_p8p8.json", "w"), indent=1); print(json.dumps(r))
PY
if python3 -c "import json,sys; sys.exit(0 if json.load(open('/workspace/logs/pair_p8p8.json'))['keep_pairing'] else 1)"; then echo "$(date -u +%T) P8 two-lane pairing KEPT"; exit 0; fi
echo "$(date -u +%T) P8 two-lane pairing NOT kept: stopping lane b and the newer job"
for p in $(ps -eo pid,args | awk '$2=="/bin/bash" && $3 ~ /m2_p8_lanes.sh$/ {print $1}'); do kill $p; done
NEW=$(ps -eo pid,etimes,args | awk -v py="$PY" '$3==py && /run_expt.py/' | sort -k2 -n | head -1 | awk '{print $1}')
A=$(ps -o args= -p $NEW); k=$(echo "$A" | sed -E 's/.*--seed ([0-9]+).*/\1/'); m=$(echo "$A" | grep -q -- "--disc" && echo disc || echo erm)
kill $(pgrep -P $NEW) $NEW; sleep 10
if [ $m = disc ]; then d=$(ls -d $D/disc/output/ISIC/DISC/*trapset_id=$k-* | head -1); else d=$D/disc/output/ISIC/ERM/*trapset_id=$k; fi
mv $d $d.stopped_pairing_$(date -u +%H%M); rmdir $D/disc/locks/${m}_$k
LANE=a nohup $Q/m2_p8_lanes.sh >> /workspace/logs/m2_p8_queue.log 2>&1 &
