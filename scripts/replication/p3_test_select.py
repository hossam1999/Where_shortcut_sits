"""Rebuild P3's (Bassi et al. 2024) o.o.d. COVID-19 test set by its stated rule (Supp. Note 5.1 and Note 10).

P3: 406 normal + 1,295 pneumonia + 1,515 COVID-19 (= 3,216; P3 states n = 3,126, a typo in one of the two). Ours (rebuilt by rule; the image lists are not published):
- normal: Montgomery and Shenzhen normals (files ending in _0; 80 + 326 = 406) aged >= 18 (age from the clinical
  readings): 370, which matches the row sums of P3's Table 4 (amendment A10).
- pneumonia: ChestX-ray14, any "Pneumonia" label, age >= 18, using NIH's Data_Entry_2017_v2020.csv (1,319 images, 938
  patients; P3: 1,295, 941). Images: the 224-px ChestX-ray14 copy (originals are 1024 x 1024, so P3's pad-to-square is a no-op).
- COVID-19: BIMCV COVID-19+ by scripts/replication/bimcv_select.py (2,501 images; P3: 1,515), as far as fetched.
Size-matched variant (amendment A9): whole patients drawn (seed 20261117) to P3's counts, 1,515 COVID-19 and 1,295
pneumonia; column `size_matched`.
Labels in P3's order: 0 normal, 1 pneumonia, 2 COVID-19. Writes $WTSS_DATA/replication/isnet/test_cohort.csv.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = f"{D}/replication/isnet/test_cohort.csv"


def note_age(png: str) -> float:
    """Age from the Montgomery/Shenzhen clinical reading next to the image ("Patient's Age: 027Y", "45yrs", "male 42").
    P3-compatible rule: the first number is read as years whatever its unit. This is the only reading that reproduces P3's
    370 adult normals (Table 4). It counts one infant as an adult: CHNCXR_0202 ("female 64days"); flagged in A10."""
    txt = open(png.replace("/CXR_png/", "/ClinicalReadings/")[:-4] + ".txt", errors="ignore").read()
    m = re.search(r"(\d+)", txt)
    return float(m.group(1)) if m else float("nan")


def main():
    rows = []
    for src, pat in (("montgomery", "cxr_at/tb/MontgomerySet/CXR_png/*_0.png"),
                     ("shenzhen", "cxr_at/tb/ChinaSet_AllFiles/CXR_png/*_0.png")):
        for f in sorted(glob.glob(f"{D}/{pat}")):
            if note_age(f) >= 18:   # amendment A10: P3's evaluated test (Table 4) has 370 adult normals
                rows.append({"image_id": os.path.basename(f)[:-4], "path": os.path.relpath(f, D), "y": 0, "source": src})
    n = pd.read_csv(f"{D}/cxr_nih_tables/data/Data_Entry_2017_v2020.csv")
    n = n[n["Finding Labels"].str.contains("Pneumonia") & (n["Patient Age"] >= 18)]
    for i in n["Image Index"]:
        rows.append({"image_id": i[:-4], "path": f"cxr_at/NIH/images-224/{i}", "y": 1, "source": "nih"})
    b = pd.read_csv(f"{D}/bimcv/selection.csv")
    for p in b.path:
        f = f"bimcv/images/{p}"
        rows.append({"image_id": os.path.basename(p)[:-4], "path": f, "y": 2, "source": "bimcv",
                     "present": os.path.exists(f"{D}/{f}")})
    t = pd.DataFrame(rows)
    # amendment A9: size-matched variant (P3's class counts), whole patients, seed 20261117
    t["patient"] = t.image_id
    t.loc[t.source == "bimcv", "patient"] = t.image_id[t.source == "bimcv"].str.extract(r"^(sub-S\d+)")[0]
    pid = dict(zip(n["Image Index"].str[:-4], n["Patient ID"].astype(str)))
    t.loc[t.source == "nih", "patient"] = "nih_" + t.image_id[t.source == "nih"].map(pid)
    rng = np.random.default_rng(20261117)
    t["size_matched"] = t.y == 0
    for src, target in (("bimcv", 1515), ("nih", 1295)):
        sizes = t[t.source == src].groupby("patient").size()
        keep, k = set(), 0
        for p in rng.permutation(sizes.index.to_numpy()):
            if k + sizes[p] <= target:
                keep.add(p); k += int(sizes[p])
            if k == target:
                break
        t.loc[(t.source == src) & t.patient.isin(keep), "size_matched"] = True
    t["present"] = t.present.fillna(True).astype(bool) if "present" in t else True
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t.to_csv(OUT, index=False)
    print(t.groupby(["y", "source"]).agg(images=("image_id", "size"), patients=("patient", "nunique"),
                                          on_disk=("present", "sum"), size_matched=("size_matched", "sum")))
    sm = t[t.size_matched]
    print("size-matched patients:", sm.groupby("source").patient.nunique().to_dict())
    print("P3 evaluated (Table 4): normal 370, pneumonia 1,295, COVID-19 1,515 (n = 3,180)")


if __name__ == "__main__":
    main()
