# Decisions log (unattended run, 2026-09-28)

Every decision taken without the author, with its reason. Commit times come from this machine and are not independent proof.

| Time (UTC) | Decision | Reason |
|---|---|---|
| 00:35 | Work on new branch `stages-and-final-runs` created from main at fb64473. | Instruction: push to a new branch, never to main. |
| 00:35 | Corrected bootstrap implemented as `wtss.stats_crossed`, selected by `WTSS_BOOTSTRAP=crossed`; default path unchanged so `make verify` still reproduces the archived pilot. | Apply the corrected estimator to every runner without editing runner scripts; keep old results reproducible. |
| 00:35 | All regenerated runs go to `results/rerun_2026-09-28/` via `WTSS_RESULTS`; predictions that already exist on this machine are copied and only statistics recomputed. | Never overwrite results; save GPU time. |
