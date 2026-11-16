"""Rebuild P3's BIMCV COVID-19+ test selection (Bassi et al. 2024, Supp. Note 5.1) from the B2DROP mirror metadata.

Rule: iterations 1+2 (5-digit subject ids; iteration 3 = 6-digit ids, added in v1.0.2, 2022); frontal (vp-ap / vp-pa)
PNG; the session's report has the NLP label "COVID 19" (not "COVID 19 uncertain"); DICOM metadata sidecar (.json) present;
age >= 18 at the session. Writes the selected images and the archives that hold them.
"""
import argparse, ast, glob, json, os, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--meta", default=os.path.expandvars("$WTSS_DATA/bimcv/meta"))
ap.add_argument("--out", default=os.path.expandvars("$WTSS_DATA/bimcv/selection.csv"))
args = ap.parse_args()
M = args.meta
X = f"{M}/x/covid19_posi"

# all files in the subject archives, from the *-tvf.txt listings
rows = []
for f in sorted(glob.glob(f"{M}/covid19_posi_subjects_part*.tar.gz-tvf.txt")):
    arch = os.path.basename(f).replace("-tvf.txt", "")
    for line in open(f, errors="ignore"):
        parts = line.split()
        if len(parts) >= 6 and not parts[-1].endswith("/"):
            rows.append((arch, int(parts[2]), parts[-1]))
files = pd.DataFrame(rows, columns=["archive", "size", "path"])
png = files[files.path.str.endswith(".png")].copy()
png["stem"] = png.path.str.replace(r"\.png$", "", regex=True)
jsons = set(files.path[files.path.str.endswith(".json")].str.replace(r"\.json$", "", regex=True))
m = png.path.str.extract(r"(sub-S(\d+))_(ses-E\d+)_run-\d+_bp-chest_vp-(\w+?)_(\w+)\.png$")
png["subject"], png["sid"], png["session"], png["view"], png["modality"] = m[0], m[1], m[2], m[3], m[4]
png = png.dropna(subset=["subject"])
n0 = len(png)
png = png[png.sid.str.len() == 5]; n_iter = len(png)
png = png[png.view.isin(["ap", "pa"])]; n_view = len(png)

L = pd.read_csv(f"{X}/derivatives/labels/Labels_covid_123.tsv", sep="\t")
def labs(s):
    try: return [str(x).strip() for x in ast.literal_eval(s)]
    except Exception: return []
L["labs"] = L.Labels.apply(labs)
covid = set(zip(L.PatientID[L.labs.apply(lambda v: "COVID 19" in v)], L.ReportID[L.labs.apply(lambda v: "COVID 19" in v)]))
png = png[[(s, e) in covid for s, e in zip(png.subject, png.session)]]; n_lab = len(png)
png = png[png.stem.isin(jsons)]; n_json = len(png)

# age: participants.tsv lists the ages over a subject's sessions; adult = youngest recorded age >= 18
P = pd.read_csv(f"{X}/participants.tsv", sep="\t")
def youngest(v):
    try: a = [x for x in ast.literal_eval(str(v)) if x is not None]
    except Exception: return float("nan")
    return float(min(a)) if a else float("nan")
age = dict(zip(P.participant, P.age.apply(youngest)))
png["age"] = png.subject.map(age)
png = png[png.age >= 18]; n_age = len(png)
png = png.drop(columns=["stem"]).sort_values("path")
png.to_csv(args.out, index=False)
counts = {"png_all": n0, "iter_1_2": n_iter, "frontal": n_view, "label_COVID19": n_lab, "dicom_json": n_json,
          "age_ge_18": n_age, "subjects": int(png.subject.nunique()), "archives": int(png.archive.nunique()),
          "archive_bytes_needed": None}
print(json.dumps(counts, indent=1))
