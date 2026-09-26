#!/bin/bash
# Download every public dataset used (≈75 GB). Idempotent. Licences: see docs/DATA.md.
set -euo pipefail
D=${WTSS_DATA:-/root/data}
mkdir -p $D/{isic2019,isic2018,artifact_masks,ham_seg,cxr/nih,cxr/neatx}
S=https://isic-challenge-data.s3.amazonaws.com
dl(){ [ -s "$2" ] || { wget -q -c -O "$2.part" "$1" && mv "$2.part" "$2"; }; }
# ISIC 2019 (CC-BY-NC) and ISIC 2018 Task 1/2 (CC-BY-NC)
dl $S/2019/ISIC_2019_Training_Input.zip $D/isic2019/ISIC_2019_Training_Input.zip &
dl $S/2019/ISIC_2019_Training_GroundTruth.csv $D/isic2019/ISIC_2019_Training_GroundTruth.csv
dl $S/2019/ISIC_2019_Training_Metadata.csv $D/isic2019/ISIC_2019_Training_Metadata.csv
dl $S/2018/ISIC2018_Task1-2_Training_Input.zip $D/isic2018/ISIC2018_Task1-2_Training_Input.zip &
dl $S/2018/ISIC2018_Task1_Training_GroundTruth.zip $D/isic2018/ISIC2018_Task1_Training_GroundTruth.zip
# HAM10000 manual lesion segmentations (Tschandl et al. 2020, Harvard Dataverse doi:10.7910/DVN/DBW86T)
dl "https://dataverse.harvard.edu/api/access/datafile/3838943" $D/ham_seg/HAM10000_segmentations_lesion_tschandl.zip
dl "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original" $D/ham_seg/HAM10000_metadata.csv
# NEATX chest-drain annotations (zenodo 14944064, CC BY-NC-SA)
for f in NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv PadChest_TubeAnnotations_NonExperts_aggregated.csv; do
  dl https://zenodo.org/api/records/14944064/files/$f/content $D/cxr/neatx/$f; done
wait
# ISIC 2019 artifact masks (Wegley et al. 2026, Scholars' Mine doi:10.71674/man1-qa33; Cloudflare-protected
# => browser-impersonating client) and NIH ChestX-ray14 (HF mirror of the official NIH release)
python - <<'PY'
import os
from curl_cffi import requests
D = os.environ.get("WTSS_DATA", "/root/data")
for k, n in {0: "vignetting.zip", 1: "hair_ruler.zip", 2: "inkmark.zip"}.items():
    p = f"{D}/artifact_masks/{n}"
    if os.path.exists(p) and os.path.getsize(p) > 0: continue
    r = requests.get(f"https://scholarsmine.mst.edu/cgi/viewcontent.cgi?filename={k}&article=1015&context=research_data&type=additional",
                     impersonate="chrome", timeout=3600, stream=True)
    with open(p, "wb") as f:
        for c in r.iter_content(chunk_size=1 << 20): f.write(c)
from huggingface_hub import hf_hub_download
files = ["data/Data_Entry_2017_v2020.csv", "data/BBox_List_2017.csv", "data/train_val_list.txt", "data/test_list.txt"] + \
        [f"data/images/images_{i:03d}.zip" for i in range(1, 13)]
for f in files:
    hf_hub_download("alkzar90/NIH-Chest-X-ray-dataset", f, repo_type="dataset", local_dir=f"{D}/cxr/nih")
PY
cd $D/cxr/nih/data/images && for z in images_*.zip; do unzip -q -n $z -d $D/cxr/nih/png; done
echo "download complete"
