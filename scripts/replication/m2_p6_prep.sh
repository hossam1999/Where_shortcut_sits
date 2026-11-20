#!/bin/bash
# A39 prerequisite: lesion U-Net (thesis script, unchanged) trained and run on machine 2 (machine 1's masks were lost), then
# the ISIC 2019 lesion ROIs as P2 (HAM10000 manual -> ISIC 2018 Task 1 -> U-Net). Beside phase 5 (gpu_wait: at most one other
# GPU job, 8 GB free; never beside ISNet, which needs 15 GB free). Log: /workspace/logs/m2_p6_prep.log (watchdog).
export WTSS_DATA=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; PY=/workspace/venv-wtss/bin/python
cd /root/Where_shortcut_sits
[ -s $WTSS_DATA/isic2019/unet/unet_resnet34_384.pt ] && grep -q heldout $WTSS_DATA/isic2019/unet/*.json 2>/dev/null || {
  echo "$(date -u +%F_%T) start unet train"
  MAX_OTHER_JOBS=1 $Q/gpu_wait.sh 8000 $PY scripts/data/train_lesion_unet.py --stage train || { echo "FAILED P6PREP unet train"; exit 1; }; }
echo "$(date -u +%F_%T) start unet predict"
MAX_OTHER_JOBS=1 $Q/gpu_wait.sh 8000 $PY scripts/data/train_lesion_unet.py --stage predict || { echo "FAILED P6PREP unet predict"; exit 1; }
echo "$(date -u +%F_%T) start lesion rois"
$PY $Q/p2_lesion_rois.py || { echo "FAILED P6PREP lesion rois"; exit 1; }
mkdir -p $WTSS_DATA/results_push/results/replication/epvt/arms && cp $WTSS_DATA/isic2019/unet/*.json $WTSS_DATA/results_push/results/replication/epvt/arms/ 2>/dev/null
echo "$(date -u +%F_%T) P6 prep done"
