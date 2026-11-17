#!/bin/bash
# Run a command only when the GPU has at least <MiB> free (so ad-hoc steps never push a training job out of memory).
# Ad-hoc steps are serialised with a lock, so two of them never pass the memory check at the same moment.
#   gpu_wait.sh <MiB> <command...>
need=$1; shift
exec 9>/tmp/wtss_gpu_adhoc.lock
flock 9
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$need" ]; do sleep 30; done
"$@"
