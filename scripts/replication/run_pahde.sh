#!/bin/bash
# One P2 (Pahde et al. 2025) run with their trainer: copies their generated config, adds only `seed` and a per-run
# checkpoint folder, disables W&B logging.   run_pahde.sh <their config yaml> <seed> <run name>
set -euo pipefail
cfg=$1; seed=$2; name=$3
R=${WTSS_DATA:-/workspace/data}/replication/pahde/runs/$name; mkdir -p $R
. /workspace/envs/pahde/bin/activate
python3 - "$cfg" "$seed" "$R" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c["seed"] = int(sys.argv[2]); c["model_savedir"] = sys.argv[3] + "/checkpoints"
yaml.safe_dump(c, open(sys.argv[3] + "/config.yaml", "w"))
PY
cd /workspace/third_party/pahde
export WANDB_MODE=disabled
exec python -u -m model_training.start_training --config_file $R/config.yaml > $R/train.log 2>&1
