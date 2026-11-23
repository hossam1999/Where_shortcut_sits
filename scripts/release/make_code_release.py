"""Export the code-only public repository (no results, data, figures, manuscripts or internal notes).

  python scripts/release/make_code_release.py /path/to/empty/target_dir

What goes in: the `wtss` package, the experiment / replication / statistics scripts, tests, third-party patches and their
SOURCE.md files, the pre-registration documents (with every amendment, as committed), the data, environment and audit
documents needed to rebuild the cohorts, and a public README (scripts/release/PUBLIC_README.md).
What stays out: results/, figures, paper and manuscript sources, reports and presentations, email drafts, machine prompts,
status notes, run-queue shell scripts with machine-specific paths, caches and anything matching a credential pattern.
The pre-registrations are copied unchanged: the "registered before results" statements in the manuscript rest on them,
including the amendments that changed a rule (e.g. A38, reproduction tolerance 0.03 -> 0.05).
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INCLUDE_DIRS = ["src", "tests", "third_party", "scripts"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "Makefile"]
INCLUDE_DOCS = [
    "STATISTICAL_PLAN.md", "THEORY.md", "DATA.md", "DATA_MANIFEST.md", "ENVIRONMENT.md", "ENVIRONMENT_MACHINE2.md",
    "ARTIFACT_MASK_AUDIT.md", "THYROID_MARKER_AUDIT.md", "SENSITIVITY_ARTIFACT_LABELS.md", "PRECOMMIT_MATCHED_TRAPS.md",
    "ROUND8_DESIGN.md", "REPLICATION_PLAN.md", "REPLICATION_SPEC.md", "P6_EPVT_AUDIT.md",
]  # plus every docs/PREREGISTRATION_*.md
EXCLUDE_SCRIPT_DIRS = {"release", "audit"}          # release handoff scripts with machine paths; rating-audit kit
EXCLUDE_NAME = re.compile(r"^(queue_.*\.sh|.*\.pyc|.*\.npz|.*\.npy|.*\.pt|.*\.ckpt|.*\.csv\.gz)$")
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", "_smoke", "_local"}
SECRET = re.compile(r"(KGAT_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                    r"|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|view_only=[0-9a-f]{8,}"
                    r"|(?i:api[_-]?key|access[_-]?token|auth[_-]?token|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,})")
MAX_BYTES = 2_000_000


def wanted(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts) or EXCLUDE_NAME.match(p.name):
        return False
    if rel.parts[0] == "scripts" and len(rel.parts) > 2 and rel.parts[1] in EXCLUDE_SCRIPT_DIRS:
        return False
    return p.is_file() and p.stat().st_size <= MAX_BYTES


def main(target: Path) -> int:
    if target.exists() and any(target.iterdir()):
        sys.exit(f"target {target} is not empty")
    files: list[Path] = []
    for d in INCLUDE_DIRS:
        files += [p for p in sorted((ROOT / d).rglob("*")) if wanted(p)]
    files += [ROOT / f for f in INCLUDE_FILES]
    files += sorted((ROOT / "docs").glob("PREREGISTRATION_*.md")) + [ROOT / "docs" / f for f in INCLUDE_DOCS]
    files.append(ROOT / "scripts" / "release" / "make_code_release.py")

    flagged = []
    for p in files:
        try:
            if SECRET.search(p.read_text(errors="ignore")):
                flagged.append(p.relative_to(ROOT))
        except OSError:
            pass
    if flagged:
        print("credential-like pattern found; nothing exported (matched text not printed):")
        for f in flagged:
            print("  ", f)
        return 1

    for p in files:
        dst = target / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    shutil.copy2(ROOT / "scripts" / "release" / "PUBLIC_README.md", target / "README.md")
    total = sum(p.stat().st_size for p in files)
    print(f"exported {len(files) + 1} files, {total / 1e6:.1f} MB, to {target}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]).resolve()))
