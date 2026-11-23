# Prompt: regenerate the real-image example figure (MMOTU + SEE-AI)

Paste everything inside the code block into Claude Code on a machine that can download the data.
- **Ovary:** CPU only, minutes.
- **Capsule:** needs DINOv2 at 518 px for the debris probe. A GPU takes about 15 minutes; on a CPU, allow a few hours.

```
Task: regenerate paper/figures/examples.{pdf,png} (Figure 2 of manuscript_bmc/manuscript.tex) for my thesis repo.
Nothing else is changed or re-run.

SETUP
1. git clone git@github.com:hossam1999/Where_shortcut_sits.git && cd Where_shortcut_sits &&
   git checkout claude/festive-bohr-95evo5
2. python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
3. export WTSS_DATA=<a data folder with ~5 GB free>  (WTSS_CACHE defaults to $WTSS_DATA/cache)

DATA (both CC BY 4.0; I will give you a Kaggle API token. Never commit it, never print it.)
4. Download only these, using the commands in scripts/data/download_extra.sh:
   - MMOTU OTU_2d: kaggle datasets download -d orvile/mmotu-ovarian-ultrasound-images-dataset -p $WTSS_DATA/ovary/mmotu --unzip
   - SEE-AI: kaggle datasets download -d capsuleyolo/kyucapsule -p $WTSS_DATA/capsule/seeai --unzip
   - Expert capsule contamination masks (figshare 27645021):
     curl -L -o $WTSS_DATA/capsule/datasets.zip https://ndownloader.figshare.com/files/50349024 && (cd $WTSS_DATA/capsule && unzip -q -o datasets.zip)
   Check the file counts against docs/DATA_MANIFEST.md (MMOTU 6,565 files; SEE-AI 36,968; masks 29,403) and report any
   difference before going on.

BUILD THE COHORTS (existing, frozen code; do not edit it)
5. PYTHONPATH=src python scripts/data/prepare_ovary.py
   Check that ovary/ovary_cohort.csv has 1,202 rows and that 31 of the 40 audited detections are calipers
   (docs/ARTIFACT_MASK_AUDIT.md); you cannot re-audit, so only report the counts the script prints.
6. PYTHONPATH=src python scripts/data/prepare_capsule.py --stage probe   (report the held-out IoU; expected about 0.62)
   PYTHONPATH=src python scripts/data/prepare_capsule.py --stage cohort  (expected 3,550 erosion + 1,931 polyp-like frames)
   If a count or the IoU differs from the expected value, stop and tell me.

FIGURE
7. PYTHONPATH=src python scripts/make_figures.py examples
   This writes paper/figures/examples.pdf, examples.png and examples_selection.csv. The selection is deterministic:
   - ovary: the longest caliper-like dotted line inside / clear of the tumour;
   - capsule: debris covering 3-20% of the frame, on / off the lesion box;
   - images in audit/sample_ids.csv are excluded.
   Do not change the rule to get a nicer picture. If the script stops with "no caliper-like ovary example", report it.
8. Look at examples.png yourself and tell me, panel by panel:
   (a) the orange mask is a caliper inside the blue tumour outline;
   (b) the orange mask is a caliper (dotted line with + ends) clearly outside the tumour, not a probe arc, screen text
       or a zoom marker;
   (c) and (d) the debris is on / off the lesion box, and the arrows point at the artifacts;
   no label or arrow covers the ROI, and the text is readable at journal size (7 inch wide).
   Send me the PNG.
9. Rebuild the manuscript in manuscript_bmc/ (pdflatex, bibtex, pdflatex, pdflatex). Check that there are 0 errors,
   0 undefined citations and 0 overfull boxes, and that Figure 2 matches its caption (blue ROI, orange masks, arrows, a-d).

RULES
- Commit only paper/figures/examples.pdf, paper/figures/examples.png, paper/figures/examples_selection.csv and
  manuscript_bmc/manuscript.pdf. Never commit data, caches, features or the Kaggle token. Never add TN3K (thyroid) or
  ISIC images to any committed figure.
- Push to claude/festive-bohr-95evo5 only. Do not create branches or pull requests.
- Report the two commands' printed counts, the chosen image IDs, the commit hash, and anything that did not match.
```

## Follow-up prompt: transplant figure (same machine, same data)

```
Follow-up on the same machine, same repo, same $WTSS_DATA (MMOTU, SEE-AI and both cohorts are already built).
Task: regenerate paper/figures/transplant_examples.{pdf,png} (Figure 3 of manuscript_bmc/manuscript.tex, also in
paper/supplement.tex). Nothing else is changed or re-run.

1. git pull origin claude/festive-bohr-95evo5  (brings the new scripts/make_transplant_figure.py)
2. Rebuild the registered placements, which are deterministic (stable seeds), for both cohorts:
     PYTHONPATH=src python scripts/run_transplant.py --cohort ovary --placements_only
     PYTHONPATH=src python scripts/run_transplant.py --cohort capsule --placements_only
   This writes $WTSS_DATA/review2/transplant_{ovary,capsule}_placements.json. It also rewrites files that are already
   committed: results/review2/transplant/<cohort>_dino518/placements.csv, design.json and feasibility_by_label.csv.
   Check with `git diff --stat results/` that these files are unchanged: same recipients, donors, ops and overlaps.
   If anything differs, stop and report the difference. Afterwards run `git checkout -- results/` so nothing under
   results/ is committed.
3. PYTHONPATH=src python scripts/make_transplant_figure.py
   This writes paper/figures/transplant_examples.{pdf,png} and transplant_selection.csv. Rows are chosen by a fixed
   rule: the two recipients per cohort with the largest donor instance. Do not change the rule.
4. Look at the PNG and tell me, row by row:
   - the pasted caliper or debris is visible, outlined in orange, and has a magnified inset;
   - "inside" is in the blue ROI and "outside" is clear of it;
   - the neutral-paste column shows the same mask shape filled with plain tissue;
   - no inset covers the ROI or the pasted region.
   Send me the PNG.
5. Rebuild manuscript_bmc (pdflatex, bibtex, pdflatex, pdflatex): 0 errors, 0 undefined citations, 0 overfull boxes.
   Rebuild paper/ only if its Makefile target works without data; otherwise skip and say so.
6. Commit only paper/figures/transplant_examples.pdf, paper/figures/transplant_examples.png,
   paper/figures/transplant_selection.csv and manuscript_bmc/manuscript.pdf. Push to claude/festive-bohr-95evo5 only.
   Never commit data, caches or tokens. Report the chosen recipients and the commit hash.
```
