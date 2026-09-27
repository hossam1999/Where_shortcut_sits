"""Text-prompted artifact directions for vision-language backbones (docs/PREREGISTRATION_TEXT_PROMPT.md).

Artifact direction(s) = differences between text embeddings of "<template> with <artifact phrase>" and "<template>"
(biased-prompt construction, cf. Chuang et al. 2023), stacked over templates × phrases; the top singular directions
(energy 0.9, max rank 8) define the erased subspace. Image features must live in the model's joint space (MedSigLIP
get_image_features, DermLIP encode_image — the cached features used throughout). Also returns class-prompt embeddings
for zero-shot classification.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "thyroid": {"templates": ["an ultrasound image", "a thyroid ultrasound image", "a sonogram of the thyroid",
                              "an ultrasound scan of a thyroid nodule"],
                "artifact": ["with measurement calipers", "with a dotted measurement line", "with cross-shaped caliper markers",
                             "with on-screen annotation marks"],
                "classes": ["an ultrasound image of a benign thyroid nodule", "an ultrasound image of a malignant thyroid nodule"]},
    "ovary": {"templates": ["an ultrasound image", "a transvaginal ultrasound image", "a sonogram of the ovary",
                            "an ultrasound scan of an ovarian mass"],
              "artifact": ["with measurement calipers", "with a dotted measurement line", "with cross-shaped caliper markers",
                           "with on-screen annotation marks"],
              "classes": ["an ultrasound image of a simple benign ovarian cyst",
                          "an ultrasound image of an ovarian tumor with solid components"]},
    "capsule": {"templates": ["a capsule endoscopy image", "a wireless capsule endoscopy frame", "an image of the small bowel",
                              "a capsule endoscopy image of the intestinal mucosa"],
                "artifact": ["with food debris", "with bubbles", "with bile and turbid fluid", "with residue covering the mucosa"],
                "classes": ["a capsule endoscopy image of a polyp", "a capsule endoscopy image of an erosion"]},
    "isic": {"templates": ["a dermoscopic image", "a dermoscopy image of a skin lesion", "a close-up photo of a skin lesion",
                           "a dermatoscopic image of a mole"],
             "artifact": ["with hair", "with hairs crossing the lesion", "with a ruler", "with dark hair strands"],
             "classes": ["a dermoscopic image of a benign nevus", "a dermoscopic image of a melanoma"]},
}


@torch.inference_mode()
def encode_text(backbone: str, texts: Sequence[str], device="cuda") -> np.ndarray:
    """L2-normalised text embeddings in the backbone's joint image–text space."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if backbone == "medsiglip448":
        from transformers import AutoModel, AutoTokenizer
        from .paths import CACHE
        kw = dict(cache_dir=str(CACHE / "hf"), token=os.environ.get("HF_TOKEN"))
        tok = AutoTokenizer.from_pretrained("google/medsiglip-448", **kw)
        m = AutoModel.from_pretrained("google/medsiglip-448", **kw).eval().to(dev)
        x = tok(list(texts), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
        out = m.get_text_features(**x)
        e = out.pooler_output if hasattr(out, "pooler_output") else out
    elif backbone == "dermlip224":
        from .backbones import _derm1m_src
        src = str(_derm1m_src().resolve())
        if src not in sys.path:
            sys.path.insert(0, src)
        import open_clip  # the Derm1M fork
        name = "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256"
        m, _, _ = open_clip.create_model_and_transforms(name)
        m = m.eval().to(dev)
        tk = open_clip.get_tokenizer(name)
        if hasattr(tk, "tokenizer") and not hasattr(tk.tokenizer, "batch_encode_plus"):
            # transformers >= 5 removed batch_encode_plus; replicate HFTokenizer.__call__ exactly
            ids = tk.tokenizer([tk.clean_fn(t) for t in texts], return_tensors="pt", max_length=tk.context_length,
                               padding="max_length", truncation=True).input_ids
            if tk.strip_sep_token:
                ids = torch.where(ids == tk.tokenizer.sep_token_id, torch.zeros_like(ids), ids)
        else:
            ids = tk(list(texts))
        e = m.encode_text(ids.to(dev))
    else:
        raise ValueError(f"{backbone} has no text encoder")
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def artifact_subspace(backbone: str, cohort: str, energy: float = 0.9, max_k: int = 8, device="cuda") -> Dict[str, np.ndarray]:
    P = PROMPTS[cohort]
    neutral = [t for t in P["templates"]]
    biased = [f"{t} {a}" for t in P["templates"] for a in P["artifact"]]
    E = encode_text(backbone, neutral + biased + P["classes"], device)
    n, b = len(neutral), len(biased)
    En, Eb, Ec = E[:n], E[n:n + b], E[n + b:]
    D = Eb - np.repeat(En, len(P["artifact"]), axis=0)  # each biased prompt minus its own template
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = int(min(max_k, np.searchsorted(cum, energy) + 1))
    return {"U": Vt[:k].T.astype(np.float32), "k": np.array(k), "class_emb": Ec.astype(np.float32),
            "diff_mean": D.mean(0).astype(np.float32)}
