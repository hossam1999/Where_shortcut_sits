# Prompt for Claude Code on a new machine: final checks before the presentation

Needs on the machine: the archive `wtss_outputs.tar` (or `.partNN` pieces, ~2.3 GB) copied to `~/archive/`, a Kaggle
token whose account accepted the RANZCR-CLiP rules, a GitHub key, ~200 GB disk, a GPU.

```
You are working on my MSc thesis repo "Where the Shortcut Sits" on a brand-new GPU machine. Goal: run the two
remaining checks before my presentation, put their results into the status report, and push. Do NOT retrain any
classifier: every prediction already exists in my archive. Only rebuild what the checks need.

SETUP
1. git clone git@github.com:hossam1999/Where_shortcut_sits.git && cd Where_shortcut_sits
   git checkout claude/festive-bohr-95evo5   (also push every commit to the branch stage-reports)
2. Python 3.12 venv; pip install -r requirements.txt (see docs/ENVIRONMENT.md). Install unrar or bsdtar, wget, and
   a LaTeX toolchain (pdflatex, bibtex, natbib).
3. export WTSS_DATA=<big disk>/data PYTHONPATH=$PWD/src WTSS_BOOTSTRAP=crossed ; KAGGLE_API_TOKEN is already set.
4. Restore predictions: the archive is at ~/archive/wtss_outputs.tar (maybe split into .partNN: cat them in order).
   Check it against SHA256SUMS / results/ARCHIVE_MANIFEST.csv, then extract it into the repo so that files land at
   results/rerun_2026-09-28/... (same paths as in the manifest). Never commit the archive or any predictions file.

DATA (only what the checks need; read docs/AFTER_RELEASE.md, Makefile targets data and data_extra, docs/DATA.md)
5. Download and prepare ISIC 2019 (+ HAM10000 masks, Wegley artifact masks), TN3K/TNCD, MMOTU, SEE-AI, NIH
   ChestX-ray14 + NEATX, and RANZCR-CLiP. Run the steps of `make data` and `make data_extra` that these need. Skip
   ISIC 2020, the round-6 external sets, ThyUS2Path, LaMa and MedGemma. Check each download against
   docs/DATA_MANIFEST.md. Do the dermoscopy/ultrasound/capsule data FIRST (check 1), the chest X-ray data second.

CHECK 1 - grouped bootstrap (most important)
6. python scripts/analysis/grouped_bootstrap.py
   It recomputes each headline interval with one weight per image (as reported) and with one weight per lesion/
   near-duplicate group. It looks for predictions in results/rerun_2026-09-28/<run>/ automatically.
7. Before trusting it, verify the IMAGE-level columns reproduce the archived numbers (the estimate exactly, the
   interval ends within about 0.005):
   skin DINOv2 +0.147 [+0.111,+0.185]; thyroid +0.239 [+0.196,+0.280]; ovary +0.170 [+0.087,+0.252];
   capsule +0.377 [+0.327,+0.426]; DermLIP +0.014 [-0.037,+0.063]; pasted-ruler sweep 518 px +0.313 [+0.258,+0.370].
   If they do not reproduce, STOP and report why; do not change the analysis to make numbers fit.
8. If the script stops with "test images have no group" (the rebuilt cohort differs slightly from the original):
   for ISIC 2019, build groups over ALL ISIC 2019 images as lesion_id ∪ pHash ≤ 8 (coarser = conservative). For the
   other cohorts, use the cohort's own group column. Write down exactly what you did in results/grouped_bootstrap/
   SUMMARY.md.

CHECK 2 - NIH patient overlap
9. python scripts/verify/check_patient_overlap.py  -> results/verification/patient_overlap.csv, must print PASS.
   If it fails, find out which experiment and how many patients, and report it; do not hide it.

REPORT UPDATE (status_reports/report1_location_of_artifacts/status_report_1.tex)
10. If all four registered DINOv2 crossovers stay significant at group level (Holm over four, in SUMMARY.md):
    remove the "preliminary" wording (summary box, "Honest error bars", Step 4 p-value sentence, Table 8) and
    replace it with one sentence such as: "With one bootstrap weight per lesion or near-duplicate group instead of
    per image, the intervals widen by a factor of X-Y and all four tests stay significant (Holm-adjusted p ≤ Z)."
    If some do not, keep the preliminary wording and report honestly which results changed and by how much.
    If check 2 passed, add a short clause that no NIH patient appears in two splits.
    Keep: easy English, max 10 pages, never the words "pilot" or "stage", no paragraph split by a float.
    Build: pdflatex, bibtex, pdflatex, pdflatex (bibtex must run before the last two passes); then delete the
    aux/bbl/blg/out/log files. Check there are no undefined citations or warnings and that it has <= 10 pages.
11. Run the test suite (pytest -q tests). Commit only small files: results/grouped_bootstrap/*,
    results/verification/patient_overlap.csv, the .tex and the .pdf. End commit messages with my usual co-author
    lines if configured. Push to claude/festive-bohr-95evo5 and to stage-reports.
12. Finish with a short plain-English summary: the grouped vs image intervals (a table), whether all four tests
    stay significant, the overlap result, and what changed in the report.
```
