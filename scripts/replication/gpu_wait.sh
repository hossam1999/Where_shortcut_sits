#!/bin/bash
# Run a command only when the GPU has at least <MiB> free AND at most MAX_OTHER_JOBS (default 1) other GPU jobs run
# (scheduling rules 2-3: never 3 GPU jobs at once unless measured faster; ISNet only alone).
# The check is serialised with a lock. A launched job leaves a pending marker ($PEND/<child pid>) that every other
# gpu_wait counts as a GPU job until the job appears in nvidia-smi or exits (2026-11-20 fix: jobs with minutes of CPU set-up,
# e.g. EPVT, used to outlast the old 180-s lock and let a third job start).
#   [MAX_OTHER_JOBS=n] gpu_wait.sh <MiB> <command...>
need=$1; shift
max=${MAX_OTHER_JOBS:-1}
PEND=/tmp/wtss_gpu_pending; mkdir -p $PEND
gpujobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do [ -d /proc/$p ] && echo $p; done | wc -l; }
pending(){ local n=0 f; for f in $PEND/*; do [ -e "$f" ] || continue; if kill -0 "$(basename $f)" 2>/dev/null; then n=$((n+1)); else rm -f "$f"; fi; done; echo $n; }
jobs(){ echo $(( $(gpujobs) + $(pending) )); }
exec 9>/tmp/wtss_gpu_adhoc2.lock
flock 9
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$need" ] && [ "$(jobs)" -le "$max" ]; do sleep 30; done
before=$(gpujobs)
"$@" 9>&- &
child=$!
touch $PEND/$child
flock -u 9; exec 9>&-
( until [ "$(gpujobs)" -gt "$before" ] || ! kill -0 $child 2>/dev/null; do sleep 5; done; rm -f $PEND/$child ) &
wait $child
