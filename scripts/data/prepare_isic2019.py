"""Prepare the ISIC 2019 real-artifact cohort (thesis Result 3).

Outputs (under $WTSS_DATA/isic2019/prepared):
  cohort.csv          one row per image: labels, source, group, mask source, artifact stats
  cache_518/          memmapped 518x518 RGB, lesion mask, hair/ruler mask (for features & inpaint)
  cache_224/          same at 224

Steps
  1. unzip images, artifact masks (Wegley et al. 2026, Scholars' Mine doi:10.71674/man1-qa33),
     HAM10000 manual lesion masks (Tschandl 2020), ISIC 2018 Task 1 manual lesion masks;
  2. leakage groups: union-find over lesion_id and pHash Hamming <= 2. The pilot used <= 8 on 2,594
     images; at ISIC 2019 scale that threshold chains unrelated lesions into one component of 10,974
     images (same-lesion precision of pHash pairs: 82% at d=0, 50% at d=2, 18% at d=4, 2% at d=8), so
     the threshold is lowered to 2 (largest group 54). Cross-fold near-duplicates are audited separately
     with DINOv2 embeddings (scripts/verify/near_duplicate_audit.py);
  3. lesion mask: manual (HAM10000 or ISIC 2018 Task 1) where available, else U-Net prediction
     (``scripts/data/train_lesion_unet.py`` must run first for those images);
  4. per-image artifact geometry at 518 px: hair fraction, hair-lesion overlap r, ink, vignetting.
"""
from __future__ import annotations

import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from wtss import paths

Image.MAX_IMAGE_PIXELS = None
D = paths.DATA


def unzip(zp: Path, dest: Path, marker: str):
    if (dest / marker).exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zp.name} -> {dest}")
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    (dest / marker).write_text("ok")


def find_file(root: Path, pattern: str) -> dict:
    return {p.name: p for p in root.rglob(pattern)}


def phash_one(p):
    import imagehash

    with Image.open(p) as im:
        im.draft("RGB", (256, 256))
        return int(str(imagehash.phash(im.convert("RGB"), hash_size=8, highfreq_factor=4)), 16)


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def leakage_groups(df: pd.DataFrame, hashes: np.ndarray, thr: int = 8):
    n = len(df)
    uf = UF(n)
    for _, idx in df.groupby("lesion_id").indices.items():
        for j in idx[1:]:
            uf.u(idx[0], j)
    h = hashes.astype(np.uint64)
    edges = 0
    for i in tqdm(range(0, n, 512), desc="pHash pairs"):
        blk = h[i:i + 512, None] ^ h[None, :]
        d = popcount64(blk)
        ii, jj = np.nonzero(d <= thr)
        for a, b in zip(ii + i, jj):
            if a < b:
                uf.u(int(a), int(b))
                edges += 1
    roots = np.array([uf.f(i) for i in range(n)])
    return roots, edges


def load_bin(p: Path | None, size: int) -> np.ndarray:
    if p is None:
        return np.zeros((size, size), np.uint8)
    with Image.open(p) as im:
        m = im.convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) >= 128).astype(np.uint8)


def per_image(args):
    """Resize image + masks to `size`; return arrays and geometry stats."""
    (image_id, img_p, lesion_p, hair_p, ink_p, vig_p, size) = args
    with Image.open(img_p) as im:
        im.draft("RGB", (size * 2, size * 2))
        rgb = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)
    lesion = load_bin(lesion_p, size) if lesion_p else None
    hair = load_bin(hair_p, size)
    ink = load_bin(ink_p, size) if ink_p else np.zeros((size, size), np.uint8)
    vig = load_bin(vig_p, size) if vig_p else np.zeros((size, size), np.uint8)
    st = {"image_id": image_id, "hair_frac": float(hair.mean()), "ink_frac": float(ink.mean()),
          "vig_frac": float(vig.mean())}
    if lesion is not None:
        inter = float((hair & lesion).sum())
        st.update(lesion_frac=float(lesion.mean()),
                  hair_in_lesion_r=inter / hair.sum() if hair.sum() else np.nan,
                  hair_cover_lesion=inter / lesion.sum() if lesion.sum() else np.nan)
    return st, rgb, lesion, hair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["unzip", "groups", "regroup", "cache"], required=True)
    ap.add_argument("--phash_threshold", type=int, default=2)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    prep = paths.ensure(D / "isic2019" / "prepared")

    if a.stage == "unzip":
        unzip(D / "isic2019" / "ISIC_2019_Training_Input.zip", D / "isic2019", ".unzipped")
        unzip(D / "isic2018" / "ISIC2018_Task1-2_Training_Input.zip", D / "isic2018", ".unzipped_img")
        unzip(D / "isic2018" / "ISIC2018_Task1_Training_GroundTruth.zip", D / "isic2018", ".unzipped_gt")
        unzip(D / "ham_seg" / "HAM10000_segmentations_lesion_tschandl.zip", D / "ham_seg", ".unzipped")
        for n in ["hair_ruler", "inkmark", "vignetting"]:
            unzip(D / "artifact_masks" / f"{n}.zip", D / "artifact_masks" / n, ".unzipped")
        return

    gt = pd.read_csv(paths.ISIC2019_GT)
    meta = pd.read_csv(paths.ISIC2019_META)
    df = gt.merge(meta, on="image").rename(columns={"image": "image_id"})
    df["y"] = (df.MEL >= 0.5).astype(int)
    df["source"] = df.lesion_id.fillna("").str.extract(r"^([A-Z]+)")[0].fillna("")
    df.loc[df.image_id.str.contains("downsampled"), "source"] = "MSK"
    df.loc[df.source == "", "source"] = "ISIC_legacy"
    df["lesion_id"] = df.lesion_id.fillna(df.image_id)
    imgs = find_file(paths.ISIC2019_IMAGES, "*.jpg")
    df["image_path"] = [str(imgs[f"{i}.jpg"]) for i in df.image_id]

    if a.stage == "regroup":  # recompute groups from saved hashes with a new threshold
        g = pd.read_csv(prep / "cohort_groups.csv")
        hs = np.array([int(x, 16) for x in g.phash_hex], dtype=np.uint64)
        roots, edges = leakage_groups(g, hs, a.phash_threshold)
        g["group"] = roots
        print(f"[groups] pHash edges<={a.phash_threshold}: {edges}; groups: {g.group.nunique()}; "
              f"largest: {g.group.value_counts().iloc[0]}")
        g.to_csv(prep / "cohort_groups.csv", index=False)
        return

    if a.stage == "groups":
        with ProcessPoolExecutor(a.workers) as ex:
            hs = list(tqdm(ex.map(phash_one, df.image_path, chunksize=64), total=len(df), desc="pHash"))
        df["phash_hex"] = [f"{h:016x}" for h in hs]
        roots, edges = leakage_groups(df, np.array(hs, dtype=np.uint64), a.phash_threshold)
        df["group"] = roots
        print(f"[groups] pHash edges<={a.phash_threshold}: {edges}; groups: {df.group.nunique()} for {len(df)} images")
        df.drop(columns=["image_path"]).to_csv(prep / "cohort_groups.csv", index=False)
        return

    # ---- cache stage
    g = pd.read_csv(prep / "cohort_groups.csv")
    df = df.merge(g[["image_id", "phash_hex", "group"]], on="image_id")
    hair = {p.name.split("_hairmask")[0]: p for p in (D / "artifact_masks" / "hair_ruler").rglob("*.tif")}
    ink = {p.name.replace("_downsample", "").split("_inkmark")[0]: p for p in (D / "artifact_masks" / "inkmark").rglob("*.tif*")}
    vig = {p.name.split("_mvig")[0]: p for p in (D / "artifact_masks" / "vignetting").rglob("*.tif")}
    ham = {p.name.split("_segmentation")[0]: p for p in paths.HAM_SEG.parent.rglob("*_segmentation.png")}
    t1 = {p.name.split("_segmentation")[0]: p for p in paths.ISIC2018_MASKS.rglob("*_segmentation.png")}
    unet_dir = D / "isic2019" / "unet_masks"
    unet = {p.name.split("_unet")[0]: p for p in unet_dir.glob("*_unet.png")} if unet_dir.exists() else {}

    def base(i):
        return i.replace("_downsampled", "")

    lesion_src, lesion_p = [], []
    for i in df.image_id:
        b = base(i)
        if b in ham:
            lesion_src.append("manual_ham"); lesion_p.append(ham[b])
        elif b in t1:
            lesion_src.append("manual_isic2018"); lesion_p.append(t1[b])
        elif i in unet:
            lesion_src.append("unet"); lesion_p.append(unet[i])
        else:
            lesion_src.append("missing"); lesion_p.append(None)
    df["lesion_mask_source"] = lesion_src
    df["has_hair_mask"] = [base(i) in hair or i in hair for i in df.image_id]
    df["has_ink_mask"] = [base(i) in ink for i in df.image_id]
    df["has_vig_mask"] = [base(i) in vig for i in df.image_id]
    print(df.lesion_mask_source.value_counts(), df[["has_hair_mask", "has_ink_mask", "has_vig_mask"]].sum())
    if (df.lesion_mask_source == "missing").any():
        print("[cache] WARNING: missing lesion masks — run train_lesion_unet.py --predict first")

    S = a.size
    cdir = paths.ensure(prep / f"cache_{S}")
    n = len(df)
    rgb = np.lib.format.open_memmap(cdir / "rgb.npy", "w+", np.uint8, (n, S, S, 3))
    les = np.lib.format.open_memmap(cdir / "roi.npy", "w+", np.uint8, (n, S, S))
    hm = np.lib.format.open_memmap(cdir / "hair.npy", "w+", np.uint8, (n, S, S))
    jobs = [(i, p, lp, hair.get(base(i), hair.get(i)), ink.get(base(i)), vig.get(base(i)), S)
            for i, p, lp in zip(df.image_id, df.image_path, lesion_p)]
    stats = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (st, r, l, h) in enumerate(tqdm(ex.map(per_image, jobs, chunksize=16), total=n, desc=f"cache {S}")):
            rgb[k] = r
            les[k] = l if l is not None else 0
            hm[k] = h
            stats.append(st)
    rgb.flush(); les.flush(); hm.flush()
    (cdir / "ids.txt").write_text("\n".join(df.image_id))
    df = df.merge(pd.DataFrame(stats), on="image_id")
    df.drop(columns=["image_path"]).to_csv(prep / f"cohort_{S}.csv", index=False)
    print(f"[cache] wrote {cdir} and cohort_{S}.csv")


if __name__ == "__main__":
    main()
