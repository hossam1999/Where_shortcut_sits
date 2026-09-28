"""Shared pieces of round 6 (docs/PREREGISTRATION_ROUND6.md, scripts/round6/README.md)."""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths

ROOT = paths.REPO_ROOT


def _load_r4():
    """Round 4's common.py is also named common; load it under its own module name."""
    path = ROOT / "scripts" / "round4" / "common.py"
    spec = importlib.util.spec_from_file_location("round4_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R4 = _load_r4()

OUT = ROOT / "results" / "round6"
FEAT = paths.CACHE / "features" / "round6"
EXT = paths.DATA / "external" / "round6"
COHORTS = ("isic", "thyroid", "capsule", "ovary")
CELLS = ("artifact_free", "trapA", "trapB", "mid", "gap")
N_BOOT_AGREE = 2000
AGREE_SEED = 20260928
A4B_SEED = 20260928


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}, default=str), flush=True)


def out_root(smoke: bool) -> Path:
    p = OUT / "_smoke" if smoke else OUT
    p.mkdir(parents=True, exist_ok=True)
    return p


def isic_key(name: str) -> str:
    m = re.search(r"ISIC_(\d+)", str(name), re.I)
    return f"ISIC_{int(m.group(1)):07d}" if m else str(name)


def coverage_gate(n: int) -> str:
    """Amendment 1: >=200 analysed, 50-199 descriptive, <50 not feasible."""
    n = int(n)
    if n >= 200:
        return "analysed"
    if n >= 50:
        return "descriptive"
    return "not_feasible"


def our_cell(free: bool, present: bool, r) -> str:
    if present:
        if pd.notna(r) and r >= 0.5:
            return "trapA"
        if pd.notna(r) and r < 0.1:
            return "trapB"
        return "mid"
    if free:
        return "artifact_free"
    return "gap"


def source_cell(present, r=None) -> str:
    """Cell implied by an independent source. r is None when the source has presence only."""
    if present is None or (isinstance(present, float) and np.isnan(present)):
        return "missing"
    if not bool(present):
        return "artifact_free"
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "present"
    if r >= 0.5:
        return "trapA"
    if r < 0.1:
        return "trapB"
    return "mid"


def contradicts(our: str, src: str) -> bool:
    """Different cell: present vs absent, or opposite sides of the ROI boundary (r>=0.5 vs r<0.1)."""
    if src in ("missing", "", None):
        return False
    presentish = {"trapA", "trapB", "mid", "present"}
    if our == "artifact_free" and src in presentish:
        return True
    if src == "artifact_free" and our in presentish:
        return True
    if our == "gap" and src in presentish:
        return True
    if {our, src} == {"trapA", "trapB"}:
        return True
    return False


def kappa_binary(a, b) -> float:
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    n = len(a)
    if n == 0:
        return float("nan")
    p0 = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1 - 1e-12:
        return float("nan")
    return (p0 - pe) / (1 - pe)


def boot_ci(stat, n: int, n_boot: int = N_BOOT_AGREE, seed: int = AGREE_SEED):
    """Image-level bootstrap. `stat(idx)` maps a resampled index array to a float."""
    if n == 0:
        return {"estimate": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0, "n_boot": 0}
    point = float(stat(np.arange(n)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.empty(n_boot, np.float64)
    for i in range(n_boot):
        vals[i] = stat(idx[i])
    vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
    return {"estimate": point, "ci95_lo": float(lo), "ci95_hi": float(hi), "n": int(n), "n_boot": int(len(vals))}


def dice_iou(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, bool).ravel()
    b = np.asarray(b, bool).ravel()
    inter = np.logical_and(a, b).sum()
    sa, sb = int(a.sum()), int(b.sum())
    union = sa + sb - inter
    dice = (2 * inter) / (sa + sb) if (sa + sb) else 1.0
    iou = inter / union if union else 1.0
    return float(dice), float(iou)


def one_sided_p(arr, alternative: str) -> float:
    """alternative 'less' tests delta<0; 'greater' tests delta>0. Floored at 1/n."""
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if alternative == "less":
        p = float((arr >= 0).mean())
    else:
        p = float((arr <= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(arr))))


def holm_table(rows: list[dict], p_key="p") -> list[dict]:
    if not rows:
        return rows
    adj = R4.holm([r[p_key] for r in rows])
    out = []
    for r, a in zip(rows, adj):
        d = dict(r)
        d["p_holm"] = float(a)
        out.append(d)
    return out


def fmt_ci(est, lo, hi) -> str:
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def original_crossover(cohort: str) -> dict:
    rel = {
        "isic": R4.REF_ROOT / "spec_e13" / "dino518_spec" / "SUMMARY.json",
        "thyroid": R4.REF_ROOT / "thyroid" / "dino518_main" / "T3_crossover.json",
        "capsule": R4.REF_ROOT / "capsule" / "dino518_main" / "T3_crossover.json",
        "ovary": R4.REF_ROOT / "ovary" / "dino518_main" / "T3_crossover.json",
    }[cohort]
    d = json.loads(rel.read_text())
    if cohort == "isic":  # crossed-bootstrap crossover of the regenerated run (Stage 3, P1)
        d = d["crossover_B_minus_A_mask"]
    return {"path": str(rel.relative_to(ROOT)), "estimate": d.get("seed_delta_mean", d.get("estimate")),
            "ci95_lo": d["ci95_lo"], "ci95_hi": d["ci95_hi"]}


def cohort_frame(name: str):
    info = R4.trap_cohort(name)
    c = info["c"].copy()
    if name == "isic":
        c["free"] = c.A0.to_numpy()
        c["present"] = (~c.A0).to_numpy()
        c["r_our"] = c.r_spec.to_numpy()
    elif name == "capsule":
        c["free"] = (c.contam_frac < 0.03).to_numpy()
        c["present"] = (c.contam_frac >= 0.10).to_numpy()
        c["r_our"] = c.r.to_numpy()
    else:
        c["free"] = (c.marker_px == 0).to_numpy()
        c["present"] = (c.marker_px >= 15).to_numpy()
        c["r_our"] = c.r.to_numpy()
    c["cell"] = [our_cell(f, p, r) for f, p, r in zip(c.free, c.present, c.r_our)]
    c["image_id"] = c.image_id.astype(str)
    info["c"] = c
    return info

def draw_contour(rgb, roi):
    """RGB image with a 2-px green contour of the ROI boundary and nothing else (no automatic artifact mask)."""
    from PIL import Image
    arr = np.array(Image.fromarray(np.asarray(rgb)).convert("RGB"))
    m = np.asarray(roi) > 0
    inner = m.copy()
    inner[1:] &= m[:-1]; inner[:-1] &= m[1:]; inner[:, 1:] &= m[:, :-1]; inner[:, :-1] &= m[:, 1:]
    b = m & ~inner
    b2 = b.copy()
    b2[1:] |= b[:-1]; b2[:-1] |= b[1:]; b2[:, 1:] |= b[:, :-1]; b2[:, :-1] |= b[:, 1:]
    arr[b2] = (0, 180, 0)
    return Image.fromarray(arr)


MEDGEMMA_ID = "google/medgemma-1.5-4b-it"
CALIPER_PROMPTS = {
    "presence": "This is an ultrasound image. Are there caliper or measurement marks overlaid on it (small plus signs, "
                "small crosses, or dotted measurement lines)? Answer only yes or no.",
    "location": "The green contour outlines the {roi}. Is any caliper or measurement mark (a small plus sign or cross) "
                "inside the green contour? Answer only yes or no.",
}


def medgemma_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return (Path.home() / ".cache" / "huggingface" / "token").exists()


class MedGemma:
    """Local MedGemma 1.5 4B-it, greedy decoding, max 3 new tokens, answer parsed as yes / no / missing."""

    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MEDGEMMA_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    def ask(self, img, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        batch = self.processor(text=text, images=img.convert("RGB"), return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=3, do_sample=False)
        ans = self.processor.decode(out[0, batch["input_ids"].shape[-1]:], skip_special_tokens=True).strip().lower()
        return "yes" if ans.startswith("yes") else "no" if ans.startswith("no") else "missing"
