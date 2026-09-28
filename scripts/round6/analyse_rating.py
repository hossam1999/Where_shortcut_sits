"""Intra-rater agreement. Runs only when both rating passes exist."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _yn(s):
    return s.map({"yes": 1, "no": 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = C.out_root(a.smoke) / "rating"
    p1, p2 = root / "pass1.csv", root / "pass2.csv"
    if not (p1.exists() and p2.exists()):
        print("analyse_rating: both passes are not saved yet; nothing to score.", flush=True)
        return
    a1, b1 = pd.read_csv(p1), pd.read_csv(p2)
    m = a1.merge(b1, on="audit_id", suffixes=("_1", "_2"))
    report = {}
    for col in ("artifact_present", "artifact_location", "mask_correct", "roi_mask_correct"):
        x, y = _yn(m[f"{col}_1"]), _yn(m[f"{col}_2"])
        ok = x.notna() & y.notna()
        report[col] = {"n": int(ok.sum()), "kappa": C.kappa_binary(x[ok].astype(int), y[ok].astype(int)) if ok.sum() else None}
    (root / "intra_rater.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
