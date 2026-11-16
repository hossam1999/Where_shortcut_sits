"""Write docs/DATA_MANIFEST.md: one row per dataset on this machine (raw downloads only; derived caches are listed in
docs/AFTER_RELEASE.md). For each: source, download date (earliest inode change time), licence, how it
was obtained, file count, total size, and the SHA256 of the sorted list "relative path<TAB>size" lines. No credentials.

  python scripts/verify/data_manifest.py
  python scripts/verify/data_manifest.py --replication   # only (re)write the replication section (docs/REPLICATION_PLAN.md)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = Path(os.environ.get("WTSS_DATA", "/root/data"))
# name, paths (relative to D), source, licence, how obtained
ROWS = [
    ("ISIC 2019 training images, ground truth, metadata", ["isic2019/ISIC_2019_Training_Input", "isic2019/ISIC_2019_Training_GroundTruth.csv", "isic2019/ISIC_2019_Training_Metadata.csv"],
     "https://isic-archive.s3.amazonaws.com/challenges/2019/", "CC BY-NC 4.0", "open"),
    ("ISIC 2018 Task 1/2 images and Task 1 masks", ["isic2018"], "https://isic-archive.s3.amazonaws.com/challenges/2018/", "CC BY-NC 4.0", "open"),
    ("HAM10000 lesion segmentations", ["ham_seg"], "https://doi.org/10.7910/DVN/DBW86T (Harvard Dataverse)", "CC BY-NC 4.0", "open"),
    ("ISIC 2019 artifact masks (Wegley et al.)", ["artifact_masks"], "https://doi.org/10.71674/man1-qa33 (Scholars' Mine)", "no licence stated on the repository page", "open"),
    ("Kabir hair masks", ["mendeley_hair"], "https://data.mendeley.com/datasets/j5ywpd2p27/2", "CC BY 4.0", "open"),
    ("ISIC 2020 labels and duplicate list (images not kept; only a derived 518-px cache remains)", ["external/isic2020"],
     "https://challenge.isic-archive.com/data (ISIC 2020 training)", "CC BY-NC 4.0", "open"),
    ("TN3K images and nodule masks with TNCD labels (thyroid)", ["us/tncd/datasets.rar", "us/tncd/ds"],
     "Google Drive archive linked from the TRFE-Net/ACL repositories (scripts/data/download_extra.sh); labels github.com/chenghui-666/ACL",
     "not stated for the data (code MIT only); images never leave this machine", "open"),
    ("MMOTU OTU_2d (ovary)", ["ovary/mmotu"], "Kaggle dataset orvile/mmotu-ovarian-ultrasound-images-dataset (mirror of figshare doi:10.6084/m9.figshare.25058690)", "CC BY 4.0", "registration (Kaggle account)"),
    ("SEE-AI capsule endoscopy dataset", ["capsule/seeai"], "Kaggle dataset capsuleyolo/kyucapsule", "CC BY 4.0", "registration (Kaggle account)"),
    ("Capsule expert contamination masks", ["capsule/datasets"], "https://ndownloader.figshare.com/files/50349024 (figshare 27645021)", "CC BY 4.0", "open"),
    ("Multicentre clear/contaminated capsule masks", ["external/round6/mendeley_capsule"], "https://data.mendeley.com/datasets/vmxhn95j8z/3", "CC BY 4.0", "open"),
    ("DermArtifactDB tables", ["external/round6/derm_tables"], "https://zenodo.org/records/18324962", "CC BY 4.0", "open"),
    ("IMA++ segmentations and metadata", ["external/round6/segs.zip", "external/round6/ima_img_metadata.csv", "external/round6/ima_seg_metadata.csv",
                                         "external/round6/img_metadata.csv", "external/round6/seg_metadata.csv",
                                         "external/round6/seg_metadata_multiannotator_subset.csv", "external/round6/iaa_metrics_image.csv",
                                         "external/round6/iaa_metrics_pairwise.csv"],
     "https://zenodo.org/records/14201693", "CC BY-NC-ND 4.0", "open"),
    ("BUSClean (code and sample data)", ["external/round6/bus-cleaning"], "https://github.com/hawaii-ai/bus-cleaning @ d2630a4", "MIT", "open"),
    ("ThyUS2Path (thyroid, Hou et al. 2024)", ["external/thyus2path"], "not recorded in the repository", "CC BY 4.0 (stages/EXTERNAL_VALIDATION_SEARCH.md)", "open"),
    ("NIH ChestX-ray14 (images and tables)", ["cxr/nih"], "Hugging Face mirror alkzar90/NIH-Chest-X-ray-dataset of the official NIH release", "NIH terms (unrestricted use, citation)", "open"),
    ("NEATX chest-drain annotations", ["cxr/neatx"], "https://zenodo.org/records/14944064", "CC BY-NC-SA", "open"),
    ("RANZCR-CLiP (training images and annotations)", ["cxr/clip/train", "cxr/clip/train.csv", "cxr/clip/train_annotations.csv"],
     "Kaggle competition ranzcr-clip-catheter-line-classification", "competition rules", "agreement (competition rules accepted on the author's Kaggle account)"),
]

# datasets added for the three-paper replication (2026-11, second machine); written to their own section
REPL_SECTION = "## Replication datasets (second machine, 2026-11)"
REPL_ROWS = [
    ("NIH ChestX-ray14 resized to 224 px, with the original Data_Entry_2017.csv (P1)", ["cxr_at/NIH"],
     "Academic Torrents e615d3aebce373f1dc8bd9d11064da55bdadede0 (the NIH/images-224 folder P1's config reads)", "NIH terms (unrestricted use, citation)", "open"),
    ("PadChest 224 px, fixed cropping, with the 160K labels CSV (P1)", ["cxr_at/PC"],
     "Academic Torrents 96ebb4f92b85929eadfb16761f310a6d04105797 (TorchXRayVision copy)", "PadChest Research Use Agreement (BIMCV)", "open"),
    ("RSNA Pneumonia Detection Challenge, JPG (P1 pneumonia)", ["cxr_at/kaggle-pneumonia-jpg"],
     "Academic Torrents 95588a735c9ae4d123f3ca408e56570409bcf2a9 (the stage_2_train_images_jpg folder P1's config reads)", "RSNA challenge terms (research use)", "open"),
    ("Montgomery County X-ray set (P3 test normals)", ["cxr_at/NLM-MontgomeryCXRSet.zip"],
     "Academic Torrents ac786f74878a5775c81d490b23842fd4736bfe33 (NLM release)", "NLM terms (public, cite)", "open"),
    ("Shenzhen Hospital X-ray set (P3 test normals)", ["cxr_at/ChinaSet_AllFiles.zip"],
     "Academic Torrents 462728e890bd37c05e9439c885df7afc36209cc8 (NLM release)", "NLM terms (public, cite)", "open"),
    ("BIMCV COVID-19+ metadata, listings, labels, SHA1 sums (P3)", ["bimcv/meta"],
     "owners' B2DROP mirror, share BIMCV-COVID19-cIter_1_2_3 (linked from github.com/BIMCV-CSUSP/BIMCV-COVID-19)", "BIMCV-COVID19 Research Use Agreement (academic, cite, no redistribution)", "open"),
    ("BIMCV COVID-19+ images selected by P3's rule (P3 test; selected PNG + JSON only)", ["bimcv/images", "bimcv/selection.csv", "bimcv/fetch_log.tsv"],
     "same mirror; archives fetched one at a time, SHA1-verified, deleted after extraction (scripts/replication/bimcv_fetch.sh)", "BIMCV-COVID19 Research Use Agreement (academic, cite, no redistribution)", "open"),
    ("ISNet pre-trained COVID-19 weights (P3)", ["isnet_weights/COVIDIsNet.pt"],
     "Google Drive folder linked from github.com/PedroRASB/ISNet README", "not stated", "open"),
    ("CheXpert-v1.0-small (P3 training normal and pneumonia; 223,414 train + 234 valid images, train.csv, valid.csv)",
     ["chexpert/CheXpert-v1.0-small"],
     "Kaggle mirror ashery/chexpert of CheXpert-v1.0-small (Kaggle API, author's account)",
     "Stanford CheXpert Research Use Agreement, accepted by the author", "registration (Kaggle account; Stanford agreement)"),
    ("BrixIA COVID-19 dataset v2, metadata (P3 training COVID-19)", ["brixia/x/data/metadata_global_v2.csv",
     "brixia/x/data/metadata_consensus_v1.csv", "brixia/x/data/changelog.txt", "brixia/x/license.txt"],
     "BrixIA, OSF, private link from the authors", "research use only, no re-identification", "private link from the authors"),
    ("BrixIA COVID-19 dataset v2, DICOM images (P3 training COVID-19; 4,695 images, 4,644 with Brixia score > 0)",
     ["brixia/dicom"], "BrixIA, OSF, private link from the authors (15 archive parts, each checked against the SHA-256 OSF reports)",
     "research use only, no re-identification", "private link from the authors"),
]


def row(name, paths, src, lic, how):
    fs = files_of(paths)
    if not fs:
        return f"| {name} | {', '.join('`'+p+'`' for p in paths)} | {src} | {lic} | {how} | — | 0 (not on disk) | 0 | — |"
    lines = sorted(f"{f.relative_to(D)}\t{f.stat().st_size}" for f in fs)
    h = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    date = dt.date.fromtimestamp(min(f.stat().st_ctime for f in fs)).isoformat()
    size = sum(f.stat().st_size for f in fs)
    return f"| {name} | {', '.join('`'+p+'`' for p in paths)} | {src} | {lic} | {how} | {date} | {len(fs)} | {size} | `{h}` |"


def replication():
    out = ROOT / "docs" / "DATA_MANIFEST.md"
    old = out.read_text()
    old = old.split("\n" + REPL_SECTION)[0].rstrip("\n") + "\n"
    L = ["", REPL_SECTION, "",
         f"Generated by `scripts/verify/data_manifest.py --replication` on {dt.date.today().isoformat()} (same columns and "
         "listing hash as above; `$WTSS_DATA=/workspace/data`). Rows still downloading show the files present at that time.", "",
         "| dataset | files on disk (under `$WTSS_DATA`) | source | licence | obtained | download date | files | size (bytes) | listing SHA256 |",
         "|---|---|---|---|---|---|---|---|---|"] + [row(*r) for r in REPL_ROWS]
    out.write_text(old + "\n".join(L) + "\n")
    print("written the replication section of docs/DATA_MANIFEST.md")


def files_of(paths):
    out = []
    for rel in paths:
        p = D / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [q for q in p.rglob("*") if q.is_file()]
    return sorted(out)


def main():
    L = ["# Data manifest (datasets on the machine that produced the results)", "",
         "Generated by `scripts/verify/data_manifest.py` on 2026-09-29. Raw downloads only; derived caches (resized images, "
         "features, LaMa-inpainted images, lung masks, audit images) are listed with their rebuild commands in "
         "`docs/AFTER_RELEASE.md`. *Listing SHA256* = SHA256 of the UTF-8 text made of one line \"relative path<TAB>size in "
         "bytes\" per file, sorted by path (relative to `$WTSS_DATA`). Download date = the date the files were created on this "
         "machine (earliest inode change time; archives keep their original modification dates, which are not used).", "",
         "| dataset | files on disk (under `$WTSS_DATA`) | source | licence | obtained | download date | files | size (bytes) | listing SHA256 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, paths, src, lic, how in ROWS:
        fs = files_of(paths)
        if not fs:
            L.append(f"| {name} | {', '.join('`'+p+'`' for p in paths)} | {src} | {lic} | {how} | — | 0 (not on disk) | 0 | — |")
            continue
        lines = sorted(f"{f.relative_to(D)}\t{f.stat().st_size}" for f in fs)
        h = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
        date = dt.date.fromtimestamp(min(f.stat().st_ctime for f in fs)).isoformat()
        size = sum(f.stat().st_size for f in fs)
        L.append(f"| {name} | {', '.join('`'+p+'`' for p in paths)} | {src} | {lic} | {how} | {date} | {len(fs)} | {size} | `{h}` |")
    (ROOT / "docs" / "DATA_MANIFEST.md").write_text("\n".join(L) + "\n")
    print("written docs/DATA_MANIFEST.md")


if __name__ == "__main__":
    import sys
    replication() if "--replication" in sys.argv else main()
