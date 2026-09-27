#!/bin/bash
# Datasets added after the pilot (thyroid, ovary, capsule, CXR devices). Idempotent. Needs:
#   KAGGLE_API_TOKEN (kaggle >= 1.7) for Kaggle datasets; RANZCR-CLiP additionally requires accepting the competition
#   rules on kaggle.com; `unrar` (or `bsdtar`) for the TN3K archive; gdown or curl for Google Drive.
set -euo pipefail
D=${WTSS_DATA:-/root/data}
kg(){ [ -d "$2" ] && return; kaggle datasets download -d "$1" -p "$2" --unzip -q; }

# Thyroid: TN3K images + nodule masks (Gong et al., TRFE-Net) with TNCD benign/malignant labels, as packaged by the
# ACL repository (https://github.com/chenghui-666/ACL; label4trainval.csv / label4test.csv).
mkdir -p $D/us/tncd/ds
if [ ! -d $D/us/tncd/ds/datasets/tn3k ]; then
  curl -sL -o $D/us/tncd/datasets.rar "https://drive.usercontent.google.com/download?id=1_BR-XfxQmJK1dPMu4F1tpJHn4syNR-i1&export=download&confirm=t"
  (cd $D/us/tncd/ds && (unrar x -o+ ../datasets.rar >/dev/null || bsdtar -xf ../datasets.rar))
fi
# Ovarian tumour ultrasound: MMOTU OTU_2d (CC BY 4.0)
kg orvile/mmotu-ovarian-ultrasound-images-dataset $D/ovary/mmotu
# Capsule endoscopy: SEE-AI (CC BY 4.0) + expert contamination masks (figshare 27645021)
kg capsuleyolo/kyucapsule $D/capsule/seeai
if [ ! -d $D/capsule/datasets ]; then
  curl -sL -o $D/capsule/datasets.zip https://ndownloader.figshare.com/files/50349024 && (cd $D/capsule && unzip -q -o datasets.zip)
fi
# Chest radiographs: RANZCR-CLiP device annotations (competition data; accept rules first)
if [ ! -d $D/cxr/clip ]; then
  mkdir -p $D/cxr/clip && kaggle competitions download -c ranzcr-clip-catheter-line-classification -p $D/cxr/clip -q \
    && (cd $D/cxr/clip && unzip -q -o ranzcr-clip-catheter-line-classification.zip)
fi
# Evaluated but not used (feasibility records in docs/): BUSI, BUS-BRA, DDTI, BKAI-IGH NeoPolyp
if [ "${WTSS_OPTIONAL:-0}" = 1 ]; then
  kg aryashah2k/breast-ultrasound-images-dataset $D/breast/busi
  kg orvile/bus-bra-a-breast-ultrasound-dataset $D/breast/busbra
  kg dasmehdixtr/ddti-thyroid-ultrasound-images $D/us/ddti
  kg dtruon46/bkai-igh-neopolyp $D/colon/bkai
fi
echo "extra datasets ready under $D"
