"""MMOTU OTU_2d ovarian-tumour ultrasound cohort + 518-px caches (gray, tumour ROI, detected caliper mask).

Task: benign cystic lesions (chocolate cyst 0, serous cystadenoma 1, simple cyst 4; y=0) vs lesions with solid
components (teratoma 2, theca cell tumour 3, mucinous cystadenoma 6, high-grade serous carcinoma 7; y=1);
normal ovary (5) excluded. Markers: frozen thyroid detector (wtss.data.us_markers.marker_mask, '+' rules).
Leakage groups: pHash Hamming <= 8 chains (no patient ids published).
"""
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths
from wtss.data.us_markers import marker_mask

D = paths.DATA / "ovary" / "mmotu" / "MMOTU" / "OTU_2d"
S = 518
Y = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1, 6: 1, 7: 1}

if __name__ == "__main__":
    rows = []
    for f in ("train_cls.txt", "val_cls.txt"):
        for l in open(D / f):
            k, v = l.split()
            if int(v) in Y:
                rows.append({"file": k, "cls": int(v), "y": Y[int(v)], "image_id": "ov_" + k.split(".")[0]})
    d = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    cdir = paths.ensure(paths.DATA / "ovary" / f"cache_{S}")
    n = len(d)
    gray = np.lib.format.open_memmap(cdir / "gray.npy", "w+", np.uint8, (n, S, S))
    roi = np.lib.format.open_memmap(cdir / "roi.npy", "w+", np.uint8, (n, S, S))
    mk = np.lib.format.open_memmap(cdir / "marker.npy", "w+", np.uint8, (n, S, S))
    mpx, rr, ph = [], [], []
    for k, f in enumerate(d.file):
        g = Image.open(D / "images" / f).convert("L")
        m = marker_mask(np.asarray(g)) > 0
        r = np.asarray(Image.open(D / "annotations" / f"{f.split('.')[0]}_binary.PNG").convert("L").resize(g.size, Image.NEAREST)) > 0
        mpx.append(int(m.sum())); rr.append(float((m & r).sum() / m.sum()) if m.sum() else np.nan)
        ph.append(str(imagehash.phash(g.convert("RGB").resize((256, 256)))))
        gray[k] = np.asarray(g.resize((S, S), Image.BICUBIC))
        roi[k] = np.asarray(Image.fromarray(r.astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) > 127
        mk[k] = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) > 127
    gray.flush(); roi.flush(); mk.flush()
    (cdir / "ids.txt").write_text("\n".join(d.image_id))
    d["marker_px"], d["r"], d["phash_hex"] = mpx, rr, ph
    H = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in ph])
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    Dm = (H[:, None, :] != H[None, :, :]).sum(-1)
    for i, j in zip(*np.nonzero(np.triu(Dm <= 8, 1))):
        a, b = find(i), find(j)
        if a != b:
            par[b] = a
    d["group"] = [find(i) for i in range(n)]
    d.to_csv(paths.DATA / "ovary" / "ovary_cohort.csv", index=False)
    g = d.group.value_counts()
    print("cached", n, "groups", len(g), "largest", g.head(3).tolist())
