"""Resumable, unattended runner for the final regeneration (docs/PREREGISTRATION_FINAL.md).

  python scripts/final/orchestrate.py <lane>      # lanes defined in scripts/final/tasks.py

Every task runs with WTSS_RESULTS=results/rerun_2026-09-28 and WTSS_BOOTSTRAP=crossed. A task that succeeds writes
results/rerun_2026-09-28/_done/<name>.ok and is skipped on restart. A failing task is retried once; if it fails again the
error is appended to stages/DECISIONS_LOG.md and the lane continues with the next task.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "final"))
from tasks import LANES, NEW  # noqa: E402

DONE = NEW / "_done"
LOG = ROOT / "stages" / "DECISIONS_LOG.md"
LOGS = NEW / "_logs"


def env():
    e = dict(os.environ)
    e.update(WTSS_RESULTS=str(NEW), WTSS_BOOTSTRAP="crossed", PYTHONPATH=str(ROOT / "src"), TQDM_DISABLE="1",
             OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    return e


def log_decision(text):
    with open(LOG, "a") as f:
        f.write(f"| {dt.datetime.utcnow():%H:%M} | {text} | Unattended rule: retry once, then record and continue. |\n")


def run_task(t) -> bool:
    name = t["name"]
    if t.get("wait_for"):
        deadline = time.time() + t.get("wait_hours", 6) * 3600
        while not Path(t["wait_for"]).exists():
            if time.time() > deadline:
                log_decision(f"Task `{name}` skipped: prerequisite `{t['wait_for']}` did not appear in time.")
                return False
            time.sleep(60)
    if t.get("copy"):
        for src, dst in t["copy"]:
            src, dst = Path(src), Path(dst)
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif not dst.exists():
                shutil.copy2(src, dst)
    if not t.get("cmd"):
        return True
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"{name}.log", "a") as f:
        f.write(f"\n=== {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S} {' '.join(t['cmd'])}\n")
        f.flush()
        r = subprocess.run(t["cmd"], cwd=ROOT, env=env(), stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def main():
    lane = sys.argv[1]
    DONE.mkdir(parents=True, exist_ok=True)
    for t in LANES[lane]:
        mark = DONE / f"{t['name']}.ok"
        if mark.exists():
            continue
        print(f"[{dt.datetime.utcnow():%H:%M:%S}] start {t['name']}", flush=True)
        ok = run_task(t) or (print(f"retry {t['name']}", flush=True) or run_task(t))
        if ok:
            mark.write_text(dt.datetime.utcnow().isoformat())
            print(f"[{dt.datetime.utcnow():%H:%M:%S}] done  {t['name']}", flush=True)
        else:
            tail = ""
            lf = LOGS / f"{t['name']}.log"
            if lf.exists():
                tail = " / ".join(l.strip() for l in lf.read_text(errors="ignore").splitlines()[-3:])[:300]
            log_decision(f"Task `{t['name']}` failed twice and was skipped (last log lines: {tail.replace('|', '/')})")
            print(f"[{dt.datetime.utcnow():%H:%M:%S}] FAILED {t['name']}", flush=True)
    print(f"lane {lane} finished", flush=True)


if __name__ == "__main__":
    main()
