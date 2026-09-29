"""Build /root/release_archive/ before the machine is released (see docs/AFTER_RELEASE.md).

  python scripts/verify/build_release_archive.py

wtss_outputs.tar (candidate for Zenodo; uncompressed — the prediction files are already gzipped; split into parts of at
most 45 GB): every results/**/predictions*.csv.gz (smoke runs excluded), every *.npz / *.npy under results/ (never the
feature caches), the weights of every trained model that produces labels or reported predictions, and the run logs of
rounds 4-9 and of the 2026-09-28/29 reruns. Every log is scanned for credentials first; a log that matches is excluded
and listed.
wtss_private_keys.tar (never into git or Zenodo): the two blinding keys and the uncommitted rating answer files. No
images. File contents are never printed.
Writes results/ARCHIVE_MANIFEST.csv (path, size, SHA256; for the private archive only file names and hashes) and
/root/release_archive/SHA256SUMS.

  python scripts/verify/build_release_archive.py --stage   # while runs are still going (run at the lowest priority)

--stage packs and hashes everything that will not change (all outputs except the still-running chest-device runs and the
logs of the rerun) into staging/, and writes the private archive. The default (final) call then appends only the new or
changed files to the staged tar and recomputes the manifest and SHA256SUMS; if a staged file changed, it rebuilds from
scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import re
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/root/release_archive")
PART = 45 * 10**9
SECRET = re.compile(r"(KGAT_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                    r"|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|(?i:api[_-]?key|access[_-]?token|auth[_-]?token|password)\s*[=:]\s*\S{8,})")
WEIGHTS = {  # archive name -> source
    "weights/isic2019_lesion_unet_spec_256.pt": "/root/data/isic2019/unet_spec/unet_spec_256.pt",
    "weights/isic2019_lesion_unet_spec_report.json": "/root/data/isic2019/unet_spec/UNET_SPEC_REPORT.json",
    "weights/isic2020_hair_unet.pt": "/root/data/external/isic2020_work/hair_unet.pt",
    "weights/capsule_debris_probe.pt": "/root/data/capsule/contam_probe.pt",
    "weights/capsule_debris_probe_eval.json": "/root/data/capsule/contam_probe_eval.json",
    "weights/drain_detector_raddino518_logreg.npz": str(OUT / "staging/weights/drain_detector_raddino518_logreg.npz"),
    "weights/DRAIN_DETECTOR.json": str(OUT / "staging/weights/DRAIN_DETECTOR.json"),
}
LOG_DIRS = ["results/round4/logs", "results/round5", "results/round6/logs", "results/round7/logs", "results/round8/logs",
            "results/round9/logs", "results/round9_search/logs", "results/rerun_2026-09-28/_logs"]
LOG_FILES = ["results/round9_search/run_search.log"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def has_secret(p: Path) -> bool:
    try:
        return bool(SECRET.search(p.read_text(errors="ignore")))
    except Exception:  # noqa: BLE001
        return True  # unreadable: exclude to be safe


def outputs() -> tuple[list, list]:
    items, excluded = [], []
    for p in sorted(ROOT.glob("results/**/predictions*.csv.gz")):
        if "_smoke" not in p.parts:
            items.append((str(p.relative_to(ROOT)), p))
    for pat in ("results/**/*.npz", "results/**/*.npy"):
        for p in sorted(ROOT.glob(pat)):
            if "_smoke" not in p.parts:
                items.append((str(p.relative_to(ROOT)), p))
    for name, src in WEIGHTS.items():
        p = Path(src)
        if p.exists():
            items.append((name, p))
        else:
            excluded.append((name, "not on disk"))
    logs = [p for d in LOG_DIRS for p in sorted((ROOT / d).rglob("*.log")) if "_smoke" not in p.parts]
    logs += [ROOT / f for f in LOG_FILES if (ROOT / f).exists()]
    for p in sorted(set(logs)):
        rel = str(p.relative_to(ROOT))
        if has_secret(p):
            excluded.append((rel, "log excluded: matches a credential pattern"))
        else:
            items.append((rel, p))
    return items, excluded


def private() -> list:
    items = []
    k1 = ROOT / "audit_local" / "KEY_open_after_review.csv"
    if k1.exists():
        items.append(("keys/" + k1.name, k1))
    for p in sorted((ROOT / "results" / "round6" / "_local").rglob("*")):
        if p.is_file() and "key" in p.name.lower() and p.suffix == ".csv":
            items.append(("keys/a4b_" + p.name, p))
    unc = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "results/round6/rating"],
                         capture_output=True, text=True).stdout.splitlines()
    for line in unc:
        p = ROOT / line[3:].strip()
        if p.is_file() and p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            items.append(("rating/" + p.name, p))
    return items


def write_tar(path: Path, items) -> list:
    if items is not None:  # None: the tar already exists (appended to a staged copy); only split it
        with tarfile.open(path, "w") as tf:
            for name, p in items:
                tf.add(str(p), arcname=name, recursive=False)
    size = path.stat().st_size
    if size <= PART:
        return [path]
    subprocess.check_call(["split", "-b", str(PART), "-d", "-a", "2", str(path), str(path) + ".part"])
    path.unlink()
    return sorted(path.parent.glob(path.name + ".part*"))


STAGE = OUT / "staging"
LIVE = ("results/rerun_2026-09-28/cxr_traps/raddino518/", "results/rerun_2026-09-28/cxr_traps/medsiglip448/",
        "results/rerun_2026-09-28/_logs/")  # still being written while --stage runs


def stage():
    STAGE.mkdir(parents=True, exist_ok=True)
    items, _ = outputs()
    items = [(n, p) for n, p in items if not n.startswith(LIVE)]
    rows = [{"path": n, "src": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime_ns, "sha256": sha256(p)} for n, p in items]
    with tarfile.open(STAGE / "outputs_stage.tar", "w") as tf:
        for n, p in items:
            tf.add(str(p), arcname=n, recursive=False)
    pd.DataFrame(rows).to_csv(STAGE / "stage_hashes.csv", index=False)
    pitems = private()
    prow = [{"path": n, "src": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime_ns, "sha256": sha256(p)} for n, p in pitems]
    write_tar(OUT / "wtss_private_keys.tar", pitems)
    pd.DataFrame(prow).to_csv(STAGE / "private_hashes.csv", index=False)
    print(f"staged {len(items)} output files, {len(pitems)} private files")


def _reuse(csv: Path, items: list):
    """Hashes of staged files that are unchanged (same size and mtime); None if any staged file changed or vanished."""
    if not csv.exists():
        return None
    st = pd.read_csv(csv)
    cur = {n: p for n, p in items}
    h = {}
    for r in st.itertuples():
        p = cur.get(r.path)
        if p is None or p.stat().st_size != r.size or p.stat().st_mtime_ns != r.mtime:
            return None
        h[r.path] = r.sha256
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true")
    if ap.parse_args().stage:
        return stage()
    OUT.mkdir(parents=True, exist_ok=True)
    items, excluded = outputs()
    pitems = private()
    staged = _reuse(STAGE / "stage_hashes.csv", items) if (STAGE / "outputs_stage.tar").exists() else None
    pstaged = _reuse(STAGE / "private_hashes.csv", pitems)
    if pstaged is not None and len(pstaged) != len(pitems):
        pstaged = None
    hashes = dict(staged or {})
    rows = [{"archive": "wtss_outputs.tar", "path": n, "size": p.stat().st_size, "sha256": hashes.get(n) or sha256(p)} for n, p in items]
    rows += [{"archive": "wtss_outputs.tar (excluded)", "path": n, "size": "", "sha256": why} for n, why in excluded]
    rows += [{"archive": "wtss_private_keys.tar", "path": n, "size": "", "sha256": (pstaged or {}).get(n) or sha256(p)} for n, p in pitems]
    for old in OUT.glob("wtss_outputs.tar*"):
        old.unlink()
    if staged is not None:
        tarp = OUT / "wtss_outputs.tar"
        shutil.copyfile(STAGE / "outputs_stage.tar", tarp)
        with tarfile.open(tarp, "a") as tf:
            for n, p in items:
                if n not in staged:
                    tf.add(str(p), arcname=n, recursive=False)
        parts = write_tar(tarp, None)
        print(f"reused staged tar ({len(staged)} files); appended {len(items) - len(staged)}")
    else:
        parts = write_tar(OUT / "wtss_outputs.tar", items)
        print("built from scratch (no usable staging)")
    pparts = [OUT / "wtss_private_keys.tar"] if pstaged is not None and (OUT / "wtss_private_keys.tar").exists() \
        else write_tar(OUT / "wtss_private_keys.tar", pitems)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "ARCHIVE_MANIFEST.csv", index=False)
    sums = [f"{sha256(p)}  {p.name}" for p in parts + pparts]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))
    print(f"outputs: {len(items)} files, {sum(p.stat().st_size for p in parts)} bytes in {len(parts)} part(s); "
          f"excluded: {len(excluded)}; private: {len(pitems)} files")
    for n, why in excluded:
        print("excluded:", n, "-", why)


if __name__ == "__main__":
    main()
