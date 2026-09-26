"""Filesystem layout. Everything large lives under WTSS_DATA; nothing is hard-coded to one machine."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = REPO_ROOT / "frozen"
RESULTS = Path(os.environ.get("WTSS_RESULTS", REPO_ROOT / "results"))
DATA = Path(os.environ.get("WTSS_DATA", "/root/data"))
CACHE = Path(os.environ.get("WTSS_CACHE", DATA / "cache"))  # resized images, features, weights

# Raw dataset locations (see scripts/download_data.sh).
ISIC2018_IMAGES = DATA / "isic2018" / "ISIC2018_Task1-2_Training_Input"
ISIC2018_MASKS = DATA / "isic2018" / "ISIC2018_Task1_Training_GroundTruth"
ISIC2019_IMAGES = DATA / "isic2019" / "ISIC_2019_Training_Input"
ISIC2019_GT = DATA / "isic2019" / "ISIC_2019_Training_GroundTruth.csv"
ISIC2019_META = DATA / "isic2019" / "ISIC_2019_Training_Metadata.csv"
HAM_SEG = DATA / "ham_seg" / "HAM10000_segmentations_lesion_tschandl"
HAM_META = DATA / "ham_seg" / "HAM10000_metadata.csv"
ARTIFACT_MASKS = DATA / "artifact_masks"
CXR_NIH = DATA / "cxr" / "nih"
NEATX = DATA / "cxr" / "neatx"


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
