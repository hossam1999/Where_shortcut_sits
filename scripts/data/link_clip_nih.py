"""Link RANZCR-CLiP images (DICOM-UID names, no disease labels) to NIH ChestX-ray14 files (disease labels,
patient ids) by 64-bit perceptual hash. CLiP images are NIH ChestX-ray14 radiographs (Tang et al. 2021).

Accept a link when the nearest NIH hash is within `max_d` bits and the second nearest is at least
`margin` bits further (unambiguous). Writes $WTSS_DATA/cxr/prepared/clip_nih_link.csv.
"""
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths

CLIP = paths.DATA / "cxr" / "clip"
NIH = paths.CXR_NIH / "png" / "images"
OUT = paths.DATA / "cxr" / "prepared"


def ph(p):
    with Image.open(p) as im:
        im.draft("L", (256, 256))
        return int(str(imagehash.phash(im.convert("L"), hash_size=8, highfreq_factor=4)), 16)


def hashes(files, cache):
    if cache.exists():
        d = pd.read_csv(cache, dtype={"h": str})
        return d.f.tolist(), np.array([int(x, 16) for x in d.h], dtype=np.uint64)
    with ProcessPoolExecutor(8) as ex:
        hs = list(ex.map(ph, files, chunksize=256))
    pd.DataFrame({"f": [str(f) for f in files], "h": [f"{h:016x}" for h in hs]}).to_csv(cache, index=False)
    return [str(f) for f in files], np.array(hs, dtype=np.uint64)


def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    nf, nh = hashes(sorted(NIH.glob("*.png")), OUT / "nih_phash.csv")
    cf, ch = hashes(sorted((CLIP / "train").glob("*.jpg")), OUT / "clip_phash.csv")
    best, second, arg = np.full(len(ch), 99), np.full(len(ch), 99), np.zeros(len(ch), int)
    for i in range(0, len(ch), 256):
        d = popcount(ch[i:i + 256, None] ^ nh[None, :])
        o = np.argpartition(d, 1, axis=1)[:, :2]
        d2 = np.take_along_axis(d, o, 1)
        s = np.argsort(d2, 1)
        o, d2 = np.take_along_axis(o, s, 1), np.take_along_axis(d2, s, 1)
        best[i:i + 256], second[i:i + 256], arg[i:i + 256] = d2[:, 0], d2[:, 1], o[:, 0]
    link = pd.DataFrame({"StudyInstanceUID": [f.split("/")[-1][:-4] for f in cf],
                         "nih_image": [nf[j].split("/")[-1] for j in arg], "d_best": best, "d_second": second})
    link["accepted"] = (link.d_best <= 6) & (link.d_second - link.d_best >= 4)
    link.to_csv(OUT / "clip_nih_link.csv", index=False)
    print(link.d_best.value_counts().sort_index().head(12).to_dict())
    print("accepted", int(link.accepted.sum()), "of", len(link), "; unique NIH targets", link[link.accepted].nih_image.nunique())
