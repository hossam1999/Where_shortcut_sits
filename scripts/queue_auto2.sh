#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
A="erm mask balanced mask_balanced mte mte_protect mte_balanced mte_protect_balanced jtt mask_jtt umte_jtt dfr mask_dfr dfr_half mask_dfr_half"
python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag auto2 --save_val --arms $A > logs/auto2_thyroid.log 2>&1
python scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag auto2 --save_val --arms $A > logs/auto2_capsule.log 2>&1
python scripts/run_thyroid_traps.py --cohort ovary --backbone dino518 --generic --tag auto2 --save_val --arms $A > logs/auto2_ovary.log 2>&1
python scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_auto2 --save_val --arms $A > logs/auto2_isic.log 2>&1
python scripts/run_drain_spec.py --backbone raddino518 --tag auto2 --save_val --arms $A > logs/auto2_drain.log 2>&1
python scripts/analysis/adaptive_select.py --split results/thyroid/dino518_auto2 results/capsule/dino518_auto2 results/ovary/dino518_auto2 results/spec_e13/dino518_spec_auto2 results/cxr_drain/raddino518_auto2 > logs/auto2_select.log 2>&1
echo DONE >> logs/auto2_select.log
