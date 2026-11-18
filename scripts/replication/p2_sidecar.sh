#!/bin/bash
# Beside the A18 fine-tunes (2026-11-18; kept only if the measured pairing raises total throughput): the GPU steps of
# p2_window.sh that do not depend on A18 — lesion U-Net train + predict, lesion ROIs (CPU), DINOv2-518 features for arms
# b, c. Same commands and outputs as p2_window.sh, which skips each of them once its output exists.
set -uo pipefail
ROOT=/root/Where_shortcut_sits; cd $ROOT
Q=$ROOT/scripts/replication; D=/workspace/data; R=$D/replication/pahde; PYW=/workspace/venv-wtss/bin/python
export WTSS_DATA=$D
log(){ echo "$(date +%F_%T) $*"; }
U=$D/isic2019/unet
[ -f $U/UNET_REPORT.json ] || PYTHONPATH=$ROOT/src $Q/gpu_wait.sh 9000 $PYW -u scripts/data/train_lesion_unet.py --stage train > $R/unet_train.log 2>&1 || { log "FAILED U-Net train"; exit 1; }
log "U-Net trained"
[ "$(ls $D/isic2019/unet_masks 2>/dev/null | wc -l)" -ge 25331 ] || PYTHONPATH=$ROOT/src $Q/gpu_wait.sh 6000 $PYW -u scripts/data/train_lesion_unet.py --stage predict > $R/unet_predict.log 2>&1 || { log "FAILED U-Net predict"; exit 1; }
[ -f $R/lesion_roi224.npz ] || CUDA_VISIBLE_DEVICES= PYTHONPATH=$ROOT/src $PYW -u $Q/p2_lesion_rois.py > $R/lesion_rois.log 2>&1 || { log "FAILED lesion ROIs"; exit 1; }
log "lesion ROIs ready"
PYTHONPATH=$ROOT/src $Q/gpu_wait.sh 6000 $PYW -u $Q/p2_arms.py --arm bc --stage features > $R/arms_bc_features.log 2>&1 || { log "FAILED arms b/c features"; exit 1; }
log "sidecar done"
