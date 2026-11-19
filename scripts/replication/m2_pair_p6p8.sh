#!/bin/bash
# Author 2026-11-19: pair the P6 A33 rerun (ERM + EPVT at bias 1, once) with P8 DISC. Measure on DISC trap set 1:
# DISC s/epoch alone (8 epochs before P6) and paired; P6 s/epoch paired and alone (DISC paused by SIGSTOP/SIGCONT, by PID incl.
# workers). Keep only if total > 1.0 of one GPU; else stop P6 by PID and rerun it after P8. Result: /workspace/logs/pair_p6p8.json
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication
DD=$(ls -d $D/disc/output/ISIC/DISC 2>/dev/null)
until ls -d $D/disc/output/ISIC/DISC/*trapset_id=1 >/dev/null 2>&1 && [ $(wc -l < $(ls -d $D/disc/output/ISIC/DISC/*trapset_id=1)/test.csv 2>/dev/null || echo 0) -ge 4 ]; do sleep 20; done
F=$(ls -d $D/disc/output/ISIC/DISC/*trapset_id=1)/test.csv
cnt(){ echo $(( $(wc -l < $F) - 1 )); }
n0=$(cnt); t0=$(date +%s); until [ $(cnt) -ge $((n0 + 8)) ]; do sleep 2; done; t1=$(date +%s); n1=$(cnt)
DA=$(python3 -c "print(($t1-$t0)/($n1-$n0))"); echo "$(date -u +%T) DISC alone ${DA} s/epoch"
for m in erm epvt; do [ -d $D/epvt/runs/${m}_bias1_1 ] && mv $D/epvt/runs/${m}_bias1_1 $D/epvt/runs/preA33_${m}_bias1_1; done
P6_START=pair BIASES=1 nohup $Q/m2_p6_queue.sh > /workspace/logs/m2_p6_a33.log 2>&1 &
P6L=$D/epvt/runs/erm_bias1_1/train.log
ep6(){ local n; n=$(grep -c "^Epoch [0-9]" $P6L 2>/dev/null); echo ${n:-0}; }
until [ "$(ep6)" -ge 1 ]; do sleep 10; done
n2=$(cnt); t2=$(date +%s); until [ "$(ep6)" -ge 6 ] && [ $(cnt) -ge $((n2 + 6)) ]; do sleep 5; done; t3=$(date +%s); n3=$(cnt)
DP=$(python3 -c "print(($t3-$t2)/($n3-$n2))")
P6P=$(grep "^Epoch [0-9]" $P6L | tail -4 | sed -E 's/.*Training time: ([0-9:]+).*/\1/' | awk -F: '{s+=$1*3600+$2*60+$3} END {print s/NR}')
MAIN=$(ps -eo pid,args | awk '$2 ~ /envs\/disc\/bin\/python/ && /run_expt.py/ && /--disc/ && /--seed 1 / {print $1}' | head -1)
KIDS=$(pgrep -P $MAIN | tr '\n' ' '); n6=$(ep6); kill -STOP $MAIN $KIDS; echo "$(date -u +%T) paused DISC $MAIN ($KIDS)"
timeout 900 bash -c "until [ \$(grep -c '^Epoch [0-9]' $P6L) -ge $((n6 + 5)) ]; do sleep 5; done"; kill -CONT $MAIN $KIDS
P6A=$(grep "^Epoch [0-9]" $P6L | tail -4 | sed -E 's/.*Training time: ([0-9:]+).*/\1/' | awk -F: '{s+=$1*3600+$2*60+$3} END {print s/NR}')
python3 - "$DA" "$DP" "$P6A" "$P6P" <<'PY'
import json, sys
da, dp, pa, pp = map(float, sys.argv[1:]); r = {"disc_s_per_epoch_alone": da, "disc_s_per_epoch_paired": dp, "disc_ratio": da / dp,
     "p6_s_per_epoch_alone": pa, "p6_s_per_epoch_paired": pp, "p6_ratio": pa / pp}; r["total"] = r["disc_ratio"] + r["p6_ratio"]; r["keep_pairing"] = r["total"] > 1.0
json.dump(r, open("/workspace/logs/pair_p6p8.json", "w"), indent=1); print(json.dumps(r))
PY
if python3 -c "import json,sys; sys.exit(0 if json.load(open('/workspace/logs/pair_p6p8.json'))['keep_pairing'] else 1)"; then echo "$(date -u +%T) pairing KEPT"; exit 0; fi
echo "$(date -u +%T) pairing NOT kept: stop P6 by PID, rerun after P8"
for p in $(ps -eo pid,args | awk '/m2_p6_queue.sh/ && !/awk/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,args | awk '$2 ~ /envs\/epvt\/bin\/python/ && /train_(erm|epvt)/ {print $1}'); do kill $(pgrep -P $p) $p 2>/dev/null; done
sleep 10; for m in erm epvt; do [ -d $D/epvt/runs/${m}_bias1_1 ] && mv $D/epvt/runs/${m}_bias1_1 $D/epvt/runs/paired_stopped_A33_${m}_bias1_1; done
nohup $Q/m2_p6_a33.sh > /workspace/logs/m2_p6_a33.log 2>&1 &
