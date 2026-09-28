# Redistribution licences of the images used in the audit sample and example figures

Checked on 2026-09-28 at the original sources. Default rule: if redistribution in a public repository is not
confirmed, images are not committed (they go to the git-ignored `audit_local/` or `figures_local/`), and only their
identifiers and the generation script are committed.

| Dataset | Licence (source) | Committed? | Attribution |
|---|---|---|---|
| ISIC 2018 Task 1/2, ISIC 2019 images (incl. HAM10000, BCN_20000, MSK) | CC BY-NC 4.0 (challenge.isic-archive.com/data) | **Yes** (raw images) | ISIC Challenge datasets 2018/2019; Tschandl et al. 2018 (HAM10000); Combalia et al. 2019 (BCN_20000); Codella et al. (MSK) |
| HAM10000 lesion segmentations | CC BY-NC 4.0 (Harvard Dataverse doi:10.7910/DVN/DBW86T) | Yes (as ROI outlines) | Tschandl et al. 2020 |
| ISIC 2019 hair/ruler masks (Wegley et al., Scholars' Mine doi:10.71674/man1-qa33) | **no licence stated** on the repository page | **No** — every overlay that draws these masks stays local | — |
| TN3K images, nodule masks; TNCD labels | **not stated for the data**; the GitHub repositories (TRFE-Net, ACL) license their *code* under MIT only | **No** — all thyroid images local | — |
| MMOTU OTU_2d | CC BY 4.0 (figshare doi:10.6084/m9.figshare.25058690) | **Yes** | Zhao et al. 2022, MMOTU |
| SEE-AI project dataset | CC BY 4.0 (Kaggle `capsuleyolo/kyucapsule`, the project's account) | **Yes** | Yokote et al. 2023/2024, SEE-AI project |
| Capsule expert contamination masks | CC BY 4.0 (figshare 27645021) | Yes | figshare 27645021 |
| NIH ChestX-ray14 | not used in the audit or example figures | — | — |

Automatic masks drawn in overlays (caliper detector, capsule probe, lesion U-Net) are our own derived data.
Note for the author: some figures committed before this check (e.g. `report/figures/data_thyroid.png`,
`report/figures/data_isic.png`) show thyroid images and ISIC hair masks; they conflict with the rule above and should
be reviewed before the repository is made more visible.
