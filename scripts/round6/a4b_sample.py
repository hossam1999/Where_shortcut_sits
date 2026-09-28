"""A4b — 150 thyroid + 150 ovary caliper images. The blind key is git-ignored; its SHA-256 is committed.

  python scripts/round6/a4b_sample.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CELLS = ("artifact_free", "trapA", "trapB")


def draw_contour(rgb, roi) -> Image.Image:
    im = Image.fromarray(np.asarray(rgb)).convert("RGB")
    m = np.asarray(roi) > 0
    # 2-px green contour of the ROI boundary
    up = np.zeros_like(m); up[1:] = m[:-1]
    dn = np.zeros_like(m); dn[:-1] = m[1:]
    lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
    boundary = m & ~(up & dn & lf & rt)
    arr = np.array(im)
    arr[boundary] = (0, 180, 0)
    # second pixel of thickness by dilating the boundary one step
    b2 = boundary.copy()
    b2[1:] |= boundary[:-1]
    b2[:-1] |= boundary[1:]
    b2[:, 1:] |= boundary[:, :-1]
    b2[:, :-1] |= boundary[:, 1:]
    arr[b2] = (0, 180, 0)
    return Image.fromarray(arr)


def sample_one(name: str, rng: np.random.Generator) -> pd.DataFrame:
    info = C.cohort_frame(name)
    c = info["c"]
    rows = []
    for cell in CELLS:
        for y in (0, 1):
            pool = c[(c.cell == cell) & (c.y == y)]
            take = min(25, len(pool))
            pick = pool.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
            rows.append(pick.assign(cohort=name))
    return pd.concat(rows, ignore_index=True), info["cache"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(C.A4B_SEED)
    local = C.ROOT / "results" / "round6" / "_local" / "a4b"
    local.mkdir(parents=True, exist_ok=True)
    pub = C.out_root(a.smoke)
    key_rows = []
    n_per = 2 if a.smoke else 25
    k = 0
    for name in ("thyroid", "ovary"):
        info = C.cohort_frame(name)
        c, cache = info["c"], info["cache"]
        for cell in CELLS:
            for y in (0, 1):
                pool = c[(c.cell == cell) & (c.y == y)]
                take = min(n_per, len(pool))
                pick = pool.sample(n=take, random_state=int(rng.integers(1, 10**9)))
                for r in pick.itertuples(index=False):
                    aid = f"B{k:04d}"
                    k += 1
                    rgb, roi, _ = cache.get(str(r.image_id))
                    raw = Image.fromarray(np.asarray(rgb))
                    over = draw_contour(rgb, roi)
                    d = local / name
                    (d / "images").mkdir(parents=True, exist_ok=True)
                    (d / "overlays").mkdir(parents=True, exist_ok=True)
                    raw.save(d / "images" / f"{aid}.png")
                    over.save(d / "overlays" / f"{aid}.png")
                    key_rows.append({"audit_id": aid, "cohort": name, "image_id": r.image_id, "y": int(r.y),
                                     "cell": cell, "image_file": f"results/round6/_local/a4b/{name}/images/{aid}.png",
                                     "overlay_file": f"results/round6/_local/a4b/{name}/overlays/{aid}.png"})
    key = pd.DataFrame(key_rows)
    key_path = local / "a4b_key.csv"
    key.to_csv(key_path, index=False)
    digest = hashlib.sha256(key_path.read_bytes()).hexdigest()
    (pub / "a4b_KEY_SHA256.txt").write_text(digest + "\n")
    blind = key[["audit_id", "cohort", "image_file", "overlay_file"]].copy()
    for col in ("artifact_present", "artifact_location", "notes"):
        blind[col] = ""
    blind.to_csv(pub / "a4b_review_sheet.csv", index=False)
    print(json.dumps({"n": len(key), "sha256": digest, "key": str(key_path)}, indent=2))


if __name__ == "__main__":
    main()
