# Shared helpers for the P8 pairing scripts (sourced). Lanes are found by the LANE=<x> in their environment.
PY=/workspace/envs/disc/bin/python; D=${WTSS_DATA:-/workspace/data}
lanes_all(){ ps -eo pid,args | awk '$2=="/bin/bash" && $3 ~ /m2_p8_lanes\.sh$/ {print $1}'; }
lane_pid(){ local p all; all=" $(lanes_all | tr '\n' ' ') "; for p in $all; do   # top-level lane shell only (not its subshells)
  [[ "$all" == *" $(ps -o ppid= -p $p | tr -d ' ') "* ]] && continue
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "LANE=$1" && echo $p; done; }
desc(){ local c; for c in $(pgrep -P $1); do echo $c; desc $c; done; }
# stop lane $1 and its running job (whole process tree); move the job's incomplete folder aside; free its lock
stop_lane(){
  local L py a k m d; L=$(lane_pid $1); [ -z "$L" ] && { echo "no lane $1"; return; }
  local T=$(desc $L)
  py=$(for p in $T; do ps -o args= -p $p 2>/dev/null | grep -q "^$PY run_expt.py" && echo $p; done | head -1)
  a=$(ps -o args= -p $py 2>/dev/null); kill $L; [ -n "$T" ] && kill $T 2>/dev/null; sleep 10
  [ -z "$a" ] && { echo "lane $1 stopped (no job)"; return; }
  k=$(echo "$a" | sed -E 's/.*--seed ([0-9]+).*/\1/'); m=$(echo "$a" | grep -q -- "--disc" && echo disc || echo erm)
  if [ $m = disc ]; then d=$(ls -d $D/disc/output/ISIC/DISC/*-trapset_id=$k | head -1); else d=$(ls -d $D/disc/output/ISIC/ERM/*-trapset_id=$k | head -1); fi
  [ -n "$d" ] && mv "$d" "$(dirname $d)/stopped_pairing_$(date -u +%H%M)_$(basename $d)_x"; rmdir $D/disc/locks/${m}_$k
  echo "lane $1 stopped; its job $m $k stopped and set aside"
}
