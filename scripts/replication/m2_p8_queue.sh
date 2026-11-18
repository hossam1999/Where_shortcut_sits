#!/bin/bash
# Machine 2, P8 step 1 (DISC, ICML 2023): their ERM then DISC on Bissoto trap sets 1-5 (their --seed = trap-set id), exact
# scripts/isic.sh commands. Order on the GPU (author): P4 -> P6 -> P8; this queue waits for the P6 bias-1 runs and for the
# concept bank, then runs alone.   nohup scripts/replication/m2_p8_queue.sh > /workspace/logs/m2_p8_queue.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; E=/workspace/third_party/disc; PY=/workspace/envs/disc/bin/python
R=$D/disc; L=$D/disc/output
until [ -s $D/epvt/runs/erm_bias1_1/erm__test_bias_1_1.json ] && [ -s $D/epvt/runs/epvt_bias1_1/epvt__test_bias_1_1.json ] \
      && [ -f $R/synthetic_concepts/.ready ]; do sleep 120; done
for k in ${SPLITS:-1 2 3 4 5}; do
  O=$L/ISIC/ERM/reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-trapset_id=$k
  if [ ! -s $O/test.csv ] || [ $(wc -l < $O/test.csv) -lt 101 ]; then
    echo "$(date -u +%F_%T) ERM trap set $k"
    (cd $E && MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 8000 $PY run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 \
      --weight_decay 1e-5 --model resnet50 --n_epochs 100 --root_dir $R/isic/ --log_dir $L/ --save_best --save_last --seed $k \
      > $D/disc/erm_$k.log 2>&1) || { echo "FAILED ERM $k"; exit 1; }
  fi
  echo "$(date -u +%F_%T) DISC trap set $k"
  (cd $E && MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 8000 $PY run_expt.py -s confounder -d ISIC -t label -c hair --lr 5e-4 --batch_size 16 \
    --weight_decay 1e-5 --model resnet50 --n_epochs 100 --root_dir $R/isic/ --log_dir $L/ --erm_path $O/best_model.pth \
    --concept_img_folder $R/synthetic_concepts --concept_categories color-texture --n_clusters 3 --save_best --save_last --seed $k --disc \
    > $D/disc/disc_$k.log 2>&1) || { echo "FAILED DISC $k"; exit 1; }
done
echo "$(date -u +%F_%T) P8 step-1 runs done"
