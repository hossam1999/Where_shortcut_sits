#!/bin/bash
# One P1 (Viviano et al. 2021) training run with their code and config, in its own working directory (own MLflow
# store and results/ folder: concurrent runs otherwise lock MLflow's SQLite database).
#   run_viviano.sh <xray|rsna> <method: resnet|resnet_maskall|resnet_actdiff|resnet_gradmask|resnet_rrr> <seed>
set -euo pipefail
task=$1; method=$2; seed=$3
V=/workspace/third_party/viviano; cfg=$V/activmask/config/${task}-seeds/${task}-seeds_${method}.yml
[ -f "$cfg" ] || { echo "no config $cfg" >&2; exit 1; }
R=${WTSS_DATA:-/workspace/data}/replication/viviano/runs/${task}_${method}_${seed}; mkdir -p $R; cd $R
. /workspace/envs/viviano/bin/activate
export MLFLOW_DISABLE_AGENT_HINT=1 TQDM_DISABLE=1 PYTHONPATH=$V:$V/activmask
exec python -u $V/activmask/main.py train --config $cfg --seed $seed > $R/train.log 2>&1
