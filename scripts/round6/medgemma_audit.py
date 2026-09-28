"""MedGemma presence and location labels for the 240-image audit.

Stops this part, without failing the rest of round 6, when the gated model cannot be used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PROMPTS = {
    "isic_hair": {
        "presence": "This is a dermoscopic image. Is there hair on the lesion? Answer only yes or no.",
        "location": "The green contour outlines the lesion. Is any hair inside the green contour? Answer only yes or no.",
    },
    "thyroid_calipers": {
        "presence": "This is an ultrasound image. Are there caliper or measurement marks overlaid on it (small plus signs, small crosses, or dotted measurement lines)? Answer only yes or no.",
        "location": "The green contour outlines the nodule. Is any caliper or measurement mark (a small plus sign or cross) inside the green contour? Answer only yes or no.",
    },
    "ovary_calipers": {
        "presence": "This is an ultrasound image. Are there caliper or measurement marks overlaid on it (small plus signs, small crosses, or dotted measurement lines)? Answer only yes or no.",
        "location": "The green contour outlines the tumour. Is any caliper or measurement mark (a small plus sign or cross) inside the green contour? Answer only yes or no.",
    },
    "capsule_debris": {
        "presence": "This is a capsule endoscopy image. Are there bubbles, debris, or turbid fluid in the image? Answer only yes or no.",
        "location": "The green contour outlines the lesion. Are there bubbles, debris, or turbid fluid inside the green contour? Answer only yes or no.",
    },
}


def token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return (Path.home() / ".cache" / "huggingface" / "token").exists()


def parse_yes_no(text: str):
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = C.out_root(a.smoke) / "rating"
    out.mkdir(parents=True, exist_ok=True)
    if not token_present():
        msg = ("MedGemma (google/medgemma-1.5-4b-it) was not run. There is no Hugging Face token on this machine. "
               "Accept the model terms at https://huggingface.co/google/medgemma-1.5-4b-it and run "
               "`huggingface-cli login`, then re-run scripts/round6/medgemma_audit.py. "
               "No audit image left this machine.")
        (out / "medgemma_status.json").write_text(json.dumps({"ran": False, "reason": msg}, indent=2))
        print(msg, flush=True)
        return
    # The token exists. Load the gated model locally and answer the fixed prompts.
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    sheet = pd.read_csv(C.ROOT / "audit" / "review_sheet.csv")
    if a.smoke:
        sheet = sheet.head(2)
    model_id = "google/medgemma-1.5-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    rows = []
    for r in sheet.itertuples(index=False):
        prompts = PROMPTS.get(r.cohort) or PROMPTS["thyroid_calipers"]
        rec = {"audit_id": r.audit_id, "cohort": r.cohort}
        for which, path, key in (("presence", r.image_file, "presence"), ("location", r.overlay_file, "location")):
            img = Image.open(C.ROOT / path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts[key]}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            batch = processor(text=text, images=img, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(**batch, max_new_tokens=3, do_sample=False)
            new = out_ids[0, batch["input_ids"].shape[-1]:]
            ans = parse_yes_no(processor.decode(new, skip_special_tokens=True))
            rec[which] = ans
        rows.append(rec)
        pd.DataFrame(rows).to_csv(out / "medgemma.csv", index=False)
        C.log(medgemma=r.audit_id, **{k: rec[k] for k in ("presence", "location")})
    (out / "medgemma_status.json").write_text(json.dumps({"ran": True, "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
