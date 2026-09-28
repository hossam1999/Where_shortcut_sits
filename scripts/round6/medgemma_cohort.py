"""A1 — MedGemma caliper labels for every thyroid and ovary image (pre-registration A1; Amendment 2).

Presence is asked on the plain image; location on the image with only the green ROI contour drawn (never our
automatic marker mask). Images in the gap (1-14 detected marker pixels) are not asked. Runs locally only.
  python scripts/round6/medgemma_cohort.py [--smoke]
Output: results/round6/matches/{thyroid,ovary}_medgemma.csv and the MedGemma rows of coverage.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROI_WORD = {"thyroid": "nodule", "ovary": "tumour"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = C.out_root(a.smoke)
    mdir = root / "matches"
    mdir.mkdir(parents=True, exist_ok=True)
    if not C.medgemma_token_present():
        (mdir / "medgemma_status.json").write_text(json.dumps({"ran": False, "reason": "no Hugging Face token; accept the "
                                                              "model terms and run huggingface-cli login"}, indent=2))
        print("MedGemma not available: no Hugging Face token. This part is stopped; the rest continues.", flush=True)
        return
    mg = C.MedGemma()
    counts = {}
    for name in ("thyroid", "ovary"):
        info = C.cohort_frame(name)
        c, cache = info["c"], info["cache"]
        c = c[c.cell != "gap"]
        if a.smoke:
            c = c.groupby("cell", group_keys=False).head(2)
        f = mdir / f"{name}_medgemma.csv"
        done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["image_id"])
        have = set(done.image_id.astype(str))
        rows = done.to_dict("records")
        for k, r in enumerate(c.itertuples(index=False)):
            if str(r.image_id) in have:
                continue
            rgb, roi, _ = cache.get(str(r.image_id))
            pres = mg.ask(Image.fromarray(np.asarray(rgb)), C.CALIPER_PROMPTS["presence"])
            loc = mg.ask(C.draw_contour(rgb, roi), C.CALIPER_PROMPTS["location"].format(roi=ROI_WORD[name]))
            rows.append({"image_id": str(r.image_id), "mg_presence": pres, "mg_location": loc})
            if k % 100 == 0:
                pd.DataFrame(rows).to_csv(f, index=False)
                C.log(medgemma=name, done=len(rows), of=len(c))
        out = pd.DataFrame(rows)
        out.to_csv(f, index=False)
        counts[name] = int((out.mg_presence != "missing").sum())
    cov_p = root / "coverage.csv"
    if cov_p.exists():
        cov = pd.read_csv(cov_p)
        for name, n in counts.items():
            m = (cov.cohort == name) & (cov.source == "MedGemma")
            cov.loc[m & (cov.cell == "all"), "n"] = n
            cov.loc[m, "gate"] = C.coverage_gate(n)
        cov.to_csv(cov_p, index=False)
    (mdir / "medgemma_status.json").write_text(json.dumps({"ran": True, "answered": counts, "model": C.MEDGEMMA_ID}, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
