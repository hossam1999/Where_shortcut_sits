"""D2: results/rerun_2026-09-28/SUMMARY_archived_reruns.md — the regeneration of the four archived analyses (A1
addendum), their input checks, the retrospective theory check, the release archive and the final checks. Every number
comes from a committed table or a log of this run; the input-check values were computed on this machine on 2026-09-29
(commands in the commit messages of the A1-addendum commits)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / "results" / "rerun_2026-09-28"
L = RR / "_logs"
ARCH = Path("/root/release_archive")


def tail(f: Path, n: int = 6) -> str:
    return "\n".join(f.read_text(errors="ignore").splitlines()[-n:]) if f.exists() else "(log missing)"


def main():
    d = pd.read_csv(ROOT / "results" / "bootstrap_correction" / "archived_old_vs_new.csv")
    fmt = lambda e, l, h: "—" if pd.isna(e) else (f"{e:+.3f}" + (f" [{l:+.3f}, {h:+.3f}]" if pd.notna(l) else ""))
    S = ["# Regeneration of the four archived analyses (A1 addendum) and release handoff", "",
         "Rule: `docs/PREREGISTRATION_FINAL.md`, A1 and its addendum (2026-09-29). Same commands, configurations and seeds; "
         "per-image predictions saved in `results/rerun_2026-09-28/`; crossed seed × image bootstrap (10,000). Full old-versus-"
         "new table: `results/bootstrap_correction/archived_old_vs_new.{csv,md}`.", "",
         "## Input checks (rebuilt inputs against the archive; nothing was adjusted or re-run to get closer)", "",
         "| analysis | rebuilt input | archived | rebuilt |", "|---|---|---|---|",
         "| LaMa | images inpainted (thyroid / ovary) | archived log no longer on disk; = images with a non-empty archived artifact mask: 1,692 / 854 | 1,692 / 854 |",
         "| LaMa | trap cell counts (thyroid, ovary) | counts.csv | identical |",
         "| capsule template | inputs | archived cached features reused | — |",
         "| chest drains | drain detector held-out precision at the registered threshold (out-of-fold, NEATX-labelled) | 0.95 | 0.9501 (1,704 of 3,543 selected; CV AUROC 0.9955) |",
         "| chest drains | real-drain cohort size | 29,687 | 29,687 |",
         "| chest drains, devices | lung masks: IoU against the archived masks still on disk (`cache/images/nih_ptx_518`, 4,076 shared images) | — | mean 1.0000, min 0.9998; 4,072 identical |",
         "| chest devices | RANZCR-CLiP images linked to NIH | 28,786 (of 30,083) | 28,786 (of 30,083) |", "",
         "## Old versus new", "", "| analysis | rows | regenerated | verdict changed | point estimate moved > 0.03 |", "|---|---|---|---|---|"]
    for an, g in d.groupby("analysis", sort=False):
        S.append(f"| {an} | {len(g)} | {int(g.new_est.notna().sum())} | {int(g.verdict_changed.fillna(False).astype(bool).sum())} | "
                 f"{int(g.est_diff_over_003.fillna(False).astype(bool).sum())} |")
    ch = d[d.verdict_changed.fillna(False).astype(bool)]
    S += ["", "Verdict changes (CI excludes zero, old → new):", ""]
    S += [f"- {r.analysis}: `{r.key}` {fmt(r.old_est, r.old_lo, r.old_hi)} → {fmt(r.new_est, r.new_lo, r.new_hi)}" for r in ch.itertuples()] or ["- none"]
    dm = d[d.file.str.contains("raddino518_devmatched", na=False)]
    missing = d[d.new_est.isna() & ~d.file.str.contains("raddino518_devmatched", na=False)]
    S += ["", f"Device-matched RAD-DINO follow-up (`cxr_traps/raddino518_devmatched`, {len(dm)} rows): **not re-estimated "
          "(dropped for time)** by the author's decision on 2026-09-29. Its archived point estimates and intervals stand "
          "unchanged and remain without per-image predictions:", ""]
    S += [f"- `{r.file.split('/')[-2]}` `{r.key}` {fmt(r.old_est, r.old_lo, r.old_hi)}" for r in dm.itertuples()] or ["- (no archived rows found)"]
    S += ["", "Not re-estimated (no regenerated value): " + (", ".join(sorted(set(missing.analysis + ": " + missing.key.str.slice(0, 60))))
                                                        if len(missing) else "none") + ".", "",
          "LaMa: the thyroid U-MtE-protect change is a reproduction difference of the protected arm (see the provenance note in "
          "`archived_old_vs_new.md`); under A1 the sentence that U-MtE-protect matches LaMa-then-mask on thyroid no longer "
          "holds and must be rewritten.", "", "## Theory (A3)", "",
          "The chest-radiograph cells enter only the retrospective comparison (`theory_with_cxr/`); the regenerated folder cited "
          "by the paper (`theory/`) and the pre-registered prospective test (84 cells, round 4) are unchanged. Registered "
          "decision rule: quantitatively predictive only if every T2 crossover sign agrees, r ≥ 0.7 for T1 and T2, and T1 MAE < "
          "reference (a).", "", "```"]
    th = L / "finish_A.log"
    S += [ln for ln in (th.read_text().splitlines() if th.exists() else []) if ln.startswith(("without chest", "with chest"))] or ["(missing)"]
    S += ["```", "", "## Preserved material (Part C)", "",
          "- C3 blinding keys: `audit_local/KEY_open_after_review.csv` **match** `audit/KEY_SHA256.txt`; the A4b key **matches** "
          "`results/round6/a4b_KEY_SHA256.txt`. `results/round6/rating/` holds 1 uncommitted file (contents not opened).",
          "- C4 archives in `/root/release_archive/` (per-file hashes: `results/ARCHIVE_MANIFEST.csv`):", "", "```"]
    S += [ln for ln in ((ARCH / "SHA256SUMS").read_text().splitlines() if (ARCH / "SHA256SUMS").exists() else [])]
    S += [f"{p.name}: {p.stat().st_size} bytes" for p in sorted(ARCH.glob("wtss_*"))]
    S += ["```", "", "Archive log (files, exclusions):", "", "```", tail(L / "C4_archive.log", 12), "```",
          "- Feature caches are not exported (5.5 GB under `$WTSS_CACHE/features`); rebuild commands: `docs/AFTER_RELEASE.md`.",
          "- Trained weights in the archive: spec lesion U-Net, ISIC 2020 hair segmenter, capsule debris probe, drain detector "
          "(parameters exported from an identical refit that reproduces the saved scores to 1e-7). Not on disk and therefore not "
          "archived: the original ISIC 2018 lesion U-Net (`unet_resnet34_384.pt`) and fine-tuned network checkpoints (the "
          "fine-tuning code never saved them). The caliper detector is rule-based (no weights).", "",
          "## Final checks (D1, D3)", "", "```", "$ make test", tail(L / "D1_make_test.log", 3),
          "$ python scripts/verify/audit_numbers_final.py --docs all", tail(L / "D1_audit_numbers.log", 6),
          "$ make verify", tail(L / "D1_make_verify.log", 6), "$ secret scan of git log -p --all", tail(L / "D3_secret_scan.log", 10),
          "```", "", "## Download (author)", "", "```bash",
          'rsync -avP -e "ssh -p <PORT>" root@<HOST>:/root/release_archive/ ./wtss_release_archive/',
          "cd wtss_release_archive && sha256sum -c SHA256SUMS", "```", ""]
    (RR / "SUMMARY_archived_reruns.md").write_text("\n".join(S) + "\n")
    s = RR / "SUMMARY.md"
    note = "\n\n**2026-09-29:** the four archived analyses without per-image predictions were regenerated — see `SUMMARY_archived_reruns.md`. " \
           "Exception: the device-matched RAD-DINO follow-up (`cxr_traps/raddino518_devmatched`) is **not re-estimated " \
           "(dropped for time)**; its archived point estimates are unchanged.\n"
    if s.exists() and "SUMMARY_archived_reruns.md" not in s.read_text():
        s.write_text(s.read_text().rstrip() + note)
    print("written", RR / "SUMMARY_archived_reruns.md")


if __name__ == "__main__":
    main()
