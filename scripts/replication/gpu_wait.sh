#!/bin/bash
# Run a command only when the GPU has at least <MiB> free AND at most MAX_OTHER_JOBS (default 1) other GPU jobs run
# (scheduling rules 2-3: never 3 GPU jobs at once unless measured faster; ISNet only beside one small job).
# Ad-hoc steps are serialised with a lock, so two of them never pass the check at the same moment.
#   [MAX_OTHER_JOBS=n] gpu_wait.sh <MiB> <command...>
need=$1; shift
max=${MAX_OTHER_JOBS:-1}
jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do [ -d /proc/$p ] && echo $p; done | wc -l; }
exec 9>/tmp/wtss_gpu_adhoc.lock
flock 9
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$need" ] && [ "$(jobs)" -le "$max" ]; do sleep 30; done
"$@"
