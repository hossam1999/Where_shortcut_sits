#!/bin/bash
# Phase-5 cohorts only (thyroid TN3K, ovary MMOTU, capsule SEE-AI + figshare contamination masks): the same sources and
# paths as scripts/data/download_extra.sh, without RANZCR. Needs KAGGLE_API_TOKEN in the environment (never in a file),
# bsdtar for the TN3K RAR. Idempotent.   WTSS_DATA=/workspace/data scripts/data/fetch_phase5.sh
set -u
D=${WTSS_DATA:?}
KG=${KAGGLE_BIN:-/workspace/venv-wtss/bin/kaggle}
log(){ echo "$(date +%F_%T) $*"; }
kg(){ [ -d "$2" ] && { log "exists: $2"; return; }; mkdir -p "$(dirname $2)"; $KG datasets download -d "$1" -p "$2" --unzip -q && log "done: $2" || log "FAILED: $1"; }
mkdir -p $D/us/tncd/ds
if [ ! -d $D/us/tncd/ds/datasets/tn3k ]; then
  [ -s $D/us/tncd/datasets.rar ] || curl -sL -o $D/us/tncd/datasets.rar "https://drive.usercontent.google.com/download?id=1_BR-XfxQmJK1dPMu4F1tpJHn4syNR-i1&export=download&confirm=t"
  (cd $D/us/tncd/ds && bsdtar -xf ../datasets.rar) && log "done: TN3K" || log "FAILED: TN3K extract"
fi
kg orvile/mmotu-ovarian-ultrasound-images-dataset $D/ovary/mmotu
kg capsuleyolo/kyucapsule $D/capsule/seeai
if [ ! -d $D/capsule/datasets ]; then
  mkdir -p $D/capsule
  curl -sL -o $D/capsule/datasets.zip https://ndownloader.figshare.com/files/50349024 && (cd $D/capsule && unzip -q -o datasets.zip) && log "done: figshare masks" || log "FAILED: figshare masks"
fi
log "fetch finished"
