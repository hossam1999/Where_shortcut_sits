#!/bin/bash
# ISNet pause for P2 (author, 2026-11-18): stop the ISNet chain cleanly at its next epoch checkpoint, run alone on the GPU
#   1. A18 RR-ClArC (λ 5e9, lr 1e-4, Vanilla weights only, fresh Adam; 3 layers, 10 epochs) + per-image scores,
#   2. lesion U-Net (train, predict all ISIC 2019), lesion ROIs (CPU),
#   3. DINOv2-518 features for P2 arms b, c,
# then restart the ISNet chain (it resumes from its checkpoint) — also on failure — and run the CPU steps.
# Each step is skipped when its output exists.   nohup scripts/replication/p2_window.sh > /workspace/logs/p2_window.log 2>&1 &
set -uo pipefail
ROOT=/root/Where_shortcut_sits; cd $ROOT
Q=scripts/replication; D=/workspace/data; R=$D/replication/pahde; P=/workspace/third_party/pahde
PYP=/workspace/envs/pahde/bin/python; PYW=/workspace/venv-wtss/bin/python
RUN=$D/replication/isnet/runs/isnet_corner_all_seed0
export WANDB_MODE=disabled WTSS_DATA=$D
log(){ echo "$(date +%F_%T) $*"; }
resume_isnet(){ pgrep -f "[p]3_chain.sh" >/dev/null || { nohup $Q/p3_chain.sh >> /workspace/logs/p3_chain.log 2>&1 & log "ISNet chain restarted"; }; }
fail(){ log "FAILED: $1"; resume_isnet; exit 1; }
alone(){ MAX_OTHER_JOBS=0 $Q/gpu_wait.sh "$@"; }

# 0. inputs prepared on CPU beforehand
until [ -f $R/render224/test_biased.csv ]; do sleep 60; done; log "renders ready"

# 1. stop the ISNet chain: controller first (so a stopped run is not taken as finished), then ISNet after its checkpoint
for p in $(pgrep -f "[p]3_chain.sh"); do kill $p && log "stopped p3_chain controller $p"; done
ISN=$(pgrep -f "[p]3_train.py --name isnet_corner_all_seed0" | while read p; do [ "$(ps -o ppid= -p $p | tr -d ' ')" != 1 ] && \
      ps -o args= -p $(ps -o ppid= -p $p) | grep -qv p3_train && echo $p; done | head -1)
if [ -n "$ISN" ]; then
  m0=$(stat -c %Y $RUN/checkpoint.pt); until [ $(stat -c %Y $RUN/checkpoint.pt) -ne $m0 ]; do sleep 5; done; sleep 3
  kill $ISN && log "ISNet $ISN stopped after epoch $(tail -1 $RUN/train_log.csv | cut -d, -f1) checkpoint"
  sleep 20
  for w in $(pgrep -f "[p]3_train.py --name isnet_corner_all_seed0"); do
    [ "$(ps -o ppid= -p $w | tr -d ' ')" = 1 ] && kill -9 $w && log "killed orphan worker $w"; done
fi
rm -f $D/replication/ISNET_CHAIN_ACTIVE

# 2. A18 RR-ClArC, alone
mkdir -p $R/scores_a18
for f in $R/mitigation_a18/*.yaml; do
  name=$(basename $f .yaml)
  [ -f $R/scores_a18/$name.csv.gz ] && continue
  [ -d $R/checkpoints_corrected_a18/$name ] && mv $R/checkpoints_corrected_a18/$name $R/checkpoints_corrected_a18/${name}_incomplete_$(date +%s)
  (cd $P && alone 6000 $PYP -u -m experiments.mitigation_experiments.start_model_correction --config_file $f > $R/mitigation_a18/$name.log 2>&1) || fail "fine-tune $name"
  alone 4000 $PYP -u $Q/p2_score.py --config $f --ckpt $R/checkpoints_corrected_a18/$name/last.ckpt --out $R/scores_a18/$name.csv.gz || fail "scoring $name"
  log "A18 $name scored"
done

# 3. lesion U-Net and ROIs
U=$D/isic2019/unet
[ -f $U/UNET_REPORT.json ] || PYTHONPATH=$ROOT/src alone 9000 $PYW -u scripts/data/train_lesion_unet.py --stage train > $R/unet_train.log 2>&1 || fail "U-Net train"
[ "$(ls $D/isic2019/unet_masks 2>/dev/null | wc -l)" -ge 25331 ] || PYTHONPATH=$ROOT/src alone 6000 $PYW -u scripts/data/train_lesion_unet.py --stage predict > $R/unet_predict.log 2>&1 || fail "U-Net predict"
[ -f $R/lesion_roi224.npz ] || CUDA_VISIBLE_DEVICES= PYTHONPATH=$ROOT/src $PYW -u $Q/p2_lesion_rois.py > $R/lesion_rois.log 2>&1 || fail "lesion ROIs"
log "lesion ROIs ready"

# 4. DINOv2-518 features for arms b, c
PYTHONPATH=$ROOT/src alone 6000 $PYW -u $Q/p2_arms.py --arm bc --stage features > $R/arms_bc_features.log 2>&1 || fail "arms b/c features"
log "GPU steps done"

# 5. ISNet back; CPU steps
resume_isnet
CUDA_VISIBLE_DEVICES= $PYW $Q/p2_select.py --set a18 > $R/select_a18.log 2>&1 || log "FAILED: A18 selection"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$ROOT/src nice -n 5 $PYW -u $Q/p2_arms.py --arm bc --stage heads > $R/arms_bc_heads.log 2>&1 || log "FAILED: arms b/c heads"
log "P2 window done"
