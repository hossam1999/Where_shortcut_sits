"""Rebuild P3's (Bassi et al. 2024) o.o.d. COVID-19 test set by its stated rule (Supp. Note 5.1 and Note 10).

P3: n = 3,126 = 406 normal + 1,295 pneumonia + 1,515 COVID-19. Ours (rebuilt by rule; the image lists are not published):
- normal: all Montgomery (80) and Shenzhen (326) normals, i.e. the files ending in _0 (406, P3's count). P3 says patients
  under 18 were removed from the other classes; this count shows the normals were kept whole (36 of the 406 are under 18).
- pneumonia: ChestX-ray14, any "Pneumonia" label, age >= 18, using NIH's Data_Entry_2017_v2020.csv (1,319 images, 938
  patients; P3: 1,295, 941). Images: the 224-px ChestX-ray14 copy (originals are 1024 x 1024, so P3's pad-to-square is a no-op).
- COVID-19: BIMCV COVID-19+ by scripts/replication/bimcv_select.py (2,501 images; P3: 1,515), as far as fetched.
Labels in P3's order: 0 normal, 1 pneumonia, 2 COVID-19. Writes $WTSS_DATA/replication/isnet/test_cohort.csv.
"""
import glob
import os

import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = f"{D}/replication/isnet/test_cohort.csv"


def main():
    rows = []
    for src, pat in (("montgomery", "cxr_at/tb/MontgomerySet/CXR_png/*_0.png"),
                     ("shenzhen", "cxr_at/tb/ChinaSet_AllFiles/CXR_png/*_0.png")):
        for f in sorted(glob.glob(f"{D}/{pat}")):
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
    t["present"] = t.present.fillna(True).astype(bool) if "present" in t else True
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t.to_csv(OUT, index=False)
    print(t.groupby(["y", "source"]).agg(images=("image_id", "size"), on_disk=("present", "sum")))
    print("P3: normal 406, pneumonia 1,295, COVID-19 1,515 (total 3,126)")


if __name__ == "__main__":
    main()
