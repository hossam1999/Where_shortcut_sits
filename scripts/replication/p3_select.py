"""Rebuild P3's (Bassi et al. 2024) COVID-19 training/validation cohort by its stated rule (Supp. Note 5.1).

Classes (P3's label order): 0 normal, 1 pneumonia, 2 COVID-19.
- COVID-19: BrixIA v2, Brixia score > 0 (4,644 images, as P3), all frontal.
- normal: CheXpert-v1.0-small train, "No Finding" = 1, frontal AP/PA, age >= 18, one image per patient, 4,644 patients drawn
  at random (P3: "4644 images of healthy patients (corresponding to 4644 different patients)").
- pneumonia: CheXpert "Pneumonia" = 1 (uncertain -1 counts as negative), frontal AP/PA, age >= 18, 4,644 images drawn at
  random from the 4,674 eligible (P3: 4,644 images from 3,899 patients).
- Split: 75% of the images train / 25% hold-out validation, by patient (no patient in both), within each class.
Seed 20261116 (registered in docs/PREREGISTRATION_REPLICATION.md, section 0). Writes $WTSS_DATA/replication/isnet/cohort.csv.
P3's own draws are not published, so this is "rebuilt by P3's rule"; the counts are printed for comparison with P3.
"""
import os

import numpy as np
import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
SEED = 20261116
OUT = f"{D}/replication/isnet/cohort.csv"


def patient_split(df, rng, frac_train=0.75):
    """P3: "75% of the samples ... for training", no patient in both. Patients are added in random order until the
    training set holds round(0.75 * n_images) images (the last patient may overshoot by its images)."""
    pats = np.array(sorted(df.patient.unique()))
    rng.shuffle(pats)
    n_img = df.patient.value_counts()
    target, tr, k = int(round(frac_train * len(df))), set(), 0
    for p in pats:
        if k >= target:
            break
        tr.add(p); k += int(n_img[p])
    return np.where(df.patient.isin(tr), "train", "valid")


def main():
    rng = np.random.default_rng(SEED)
    b = pd.read_csv(f"{D}/brixia/x/data/metadata_global_v2.csv", sep=";")
    b = b[b.BrixiaScoreGlobal > 0]
    covid = pd.DataFrame({"image_id": b.Filename.str.replace(".dcm", "", regex=False),
                          "path": "brixia/dicom/dicom_clean/" + b.Filename, "patient": "brixia_" + b.Subject,
                          "view": "", "y": 2, "source": "brixia"})
    c = pd.read_csv(f"{D}/chexpert/CheXpert-v1.0-small/train.csv")
    c["patient"] = c.Path.str.extract(r"(patient\d+)")[0]
    c = c[(c["Frontal/Lateral"] == "Frontal") & c["AP/PA"].isin(["AP", "PA"]) & (c.Age >= 18)]
    nf = c[c["No Finding"] == 1]
    nf = nf.loc[rng.permutation(nf.index.to_numpy())].drop_duplicates("patient")  # one random image per patient
    nf = nf.loc[rng.choice(nf.index.to_numpy(), 4644, replace=False)]
    pn = c[c["Pneumonia"] == 1]
    pn = pn.loc[rng.choice(pn.index.to_numpy(), 4644, replace=False)]
    parts = [covid]
    for df, y in ((nf, 0), (pn, 1)):
        parts.append(pd.DataFrame({"image_id": df.Path.str.replace("/", "_").str.replace(".jpg", "", regex=False),
                                   "path": "chexpert/" + df.Path, "patient": "chexpert_" + df.patient,
                                   "view": df["AP/PA"], "y": y, "source": "chexpert"}))
    out = []
    for p in parts:
        p = p.copy()
        p["split"] = patient_split(p, rng)
        out.append(p)
    co = pd.concat(out).sort_values(["y", "image_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    co.to_csv(OUT, index=False)
    print(co.groupby(["y", "split"]).agg(images=("image_id", "size"), patients=("patient", "nunique")))
    print("P3: COVID-19 train 3,483 images; each class 75/25 by patient")


if __name__ == "__main__":
    main()
