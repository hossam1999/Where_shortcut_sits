"""A4 — MedGemma presence and location labels for the 240-image audit (pre-registration A4).

Presence is asked on the raw audit image; location on the image with only the green ROI contour drawn from the
cohort cache (never on the audit overlay, which also shows our automatic artifact mask). Runs locally only; stops
this part, without failing the rest of round 6, when the gated model cannot be used.
  python scripts/round6/medgemma_audit.py [--smoke]
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

COHORT = {"isic_hair": "isic", "thyroid_calipers": "thyroid", "ovary_calipers": "ovary", "capsule_debris": "capsule"}
PROMPTS = {
    "isic_hair": {
        "presence": "This is a dermoscopic image. Is there hair visible anywhere in the image? Answer only yes or no.",
        "location": "The green contour outlines the lesion. Is any hair inside the green contour? Answer only yes or no.",
    },
    "thyroid_calipers": {"presence": C.CALIPER_PROMPTS["presence"],
                         "location": C.CALIPER_PROMPTS["location"].format(roi="nodule")},
    "ovary_calipers": {"presence": C.CALIPER_PROMPTS["presence"],
                       "location": C.CALIPER_PROMPTS["location"].format(roi="tumour")},
    "capsule_debris": {
        "presence": "This is a capsule endoscopy image. Are there bubbles, debris, or turbid fluid in the image? "
                    "Answer only yes or no.",
        "location": "The green contour outlines the lesion area. Are there bubbles, debris, or turbid fluid inside the "
                    "green contour? Answer only yes or no.",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = C.out_root(a.smoke) / "rating"
    out.mkdir(parents=True, exist_ok=True)
    if not C.medgemma_token_present():
        msg = (f"MedGemma ({C.MEDGEMMA_ID}) was not run: no Hugging Face token on this machine. Accept the model terms "
               "on its Hugging Face page, run `huggingface-cli login`, then re-run scripts/round6/medgemma_audit.py. "
               "No audit image left this machine.")
        (out / "medgemma_status.json").write_text(json.dumps({"ran": False, "reason": msg}, indent=2))
        print(msg, flush=True)
        return
    sheet = pd.read_csv(C.ROOT / "audit" / "review_sheet.csv")
    ids = pd.read_csv(C.ROOT / "audit" / "sample_ids.csv")  # audit_id -> cohort image_id (no cell information)
    sheet = sheet.merge(ids[["audit_id", "image_id"]], on="audit_id", how="left")
    if a.smoke:
        sheet = sheet.groupby("cohort", group_keys=False).head(1)
    caches = {}
    mg = C.MedGemma()
    rows = []
    for r in sheet.itertuples(index=False):
        name = COHORT[r.cohort]
        if name not in caches:
            caches[name] = C.cohort_frame(name)["cache"]
        rgb, roi, _ = caches[name].get(str(r.image_id))
        pres = mg.ask(Image.open(C.ROOT / r.image_file), PROMPTS[r.cohort]["presence"])
        loc = mg.ask(C.draw_contour(np.asarray(rgb), roi), PROMPTS[r.cohort]["location"])
        rows.append({"audit_id": r.audit_id, "cohort": r.cohort, "presence": pres, "location": loc})
        pd.DataFrame(rows).to_csv(out / "medgemma.csv", index=False)
        C.log(medgemma=r.audit_id, presence=pres, location=loc)
    (out / "medgemma_status.json").write_text(json.dumps({"ran": True, "n": len(rows), "model": C.MEDGEMMA_ID}, indent=2))


if __name__ == "__main__":
    main()
