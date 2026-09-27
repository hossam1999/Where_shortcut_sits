#!/bin/bash
# GPU part first (text encoders; ovary MedSigLIP features), then CPU fits after the SPLICE re-run finishes.
cd /root/wtss
source /venv/main/bin/activate
source /root/.config/wtss/secrets.env
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
python scripts/make_text_directions.py > logs/text_dirs.log 2>&1
python scripts/run_thyroid_traps.py --cohort ovary --backbone medsiglip448 --generic --tag feat --arms erm mte > logs/ovary_medsiglip_feat.log 2>&1
python scripts/analysis/zero_shot.py > logs/zero_shot.log 2>&1
until grep -q DONE logs/splice2_drain.log 2>/dev/null; do sleep 30; done
A="erm mask balanced mte mte_protect mte_balanced text_erase mask_text_erase mask_text_erase_balanced mask_text_erase_protect"
for c in thyroid capsule ovary; do
  python scripts/run_thyroid_traps.py --cohort $c --backbone medsiglip448 --generic --tag text --text_dirs results/text_dirs/medsiglip448_$c.npz --arms $A > logs/text_$c.log 2>&1
done
python scripts/run_spec_e13.py --backbone dermlip224 --generic --tag spec_text --text_dirs results/text_dirs/dermlip224_isic.npz --arms $A > logs/text_isic.log 2>&1
echo DONE >> logs/text_isic.log
