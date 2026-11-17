#!/bin/bash
# Run a command only when the GPU has at least <MiB> free (so ad-hoc steps never push a training job out of memory).
#   gpu_wait.sh <MiB> <command...>
need=$1; shift
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$need" ]; do sleep 30; done
exec "$@"
