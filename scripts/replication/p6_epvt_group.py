"""P6 (EPVT, Yan et al. MICCAI 2023 / PLDG, TMI 2025) data: their data_proc/grouping.py, run unchanged in logic for one
Bissoto trap set (artifact-generalization-skin @ 54b20fb). Their processed Google Drive archive (EPVT README) returns 404
(2026-11-18), so the training folder is rebuilt here. Changes: paths only (their server paths -> ours), the trap-set CSV
(their grouping reads train_bias_0_1.csv; the trap-set experiment needs train_bias_<b>_<k>.csv, the paper does not say which
split k, we use k = 1), images symlinked instead of copied (same bytes), and the domain folder names mapped to the letters
their SKIN dataset class reads (ENVIRONMENTS C, D, G, H, R = clean, dark_corner, gel_bubble, hair, ruler).

    python scripts/replication/p6_epvt_group.py --bias 1 --split 1
Output: $WTSS_DATA/epvt/bias<b>_<k>/ISIC2019_train/{C,D,G,H,R}/{ben,mel}/*.jpg and counts.json
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
B = "/workspace/third_party/bissoto_ag"
ISIC = f"{D}/epvt/isic2019_plain/"   # ISIC 2019 with "_downsampled" stripped from names (Bissoto CSVs use plain ids)
LETTER = {"clean": "C", "dark_corner": "D", "gel_bubble": "G", "hair": "H", "ruler": "R"}

ap = argparse.ArgumentParser(); ap.add_argument("--bias", required=True); ap.add_argument("--split", type=int, default=1)
a = ap.parse_args()
root_csv = f"{B}/trap_sets/train_bias_{a.bias}_{a.split}.csv"
# ---- their grouping.py logic, verbatim apart from paths ----
attrs_df = pd.read_csv(f"{B}/isic_inferred_wocarcinoma.csv")
data = pd.read_csv(root_csv)
attrs_df = attrs_df[attrs_df['image'].isin(data['image'])]
env_df = attrs_df.drop(labels='image', axis='columns')
env_df = env_df.drop(labels='Unnamed: 0', axis='columns')
env_df = env_df.drop(labels='vasc', axis='columns')
env_df = env_df.drop(labels='label', axis='columns')
env_df = env_df.drop(labels='label_string', axis='columns')
env_df = env_df.drop(labels='gel_border', axis='columns')
env_df = env_df.drop(labels='ink', axis='columns')
env_df = env_df.drop(labels='patches', axis='columns')
attrs_df = attrs_df.copy()
attrs_df['max_prob'] = env_df.max(axis=1)
attrs_df.loc[attrs_df['max_prob'] > 0.65, 'max_prob'] = 0          # their chained assignment, same effect
attrs_df.loc[(attrs_df['max_prob'] <= 0.65) & (attrs_df['max_prob'] > 0), 'max_prob'] = 1
attrs_df.rename(columns={'max_prob': 'clean'}, inplace=True)
df_final = pd.concat([env_df, attrs_df['clean']], axis=1)
env_df = df_final.mask(df_final.eq(df_final.max(axis=1), axis=0), 1, axis=0)
df_final = pd.concat([attrs_df['image'], env_df, attrs_df['label']], axis=1)
# ---- copy step (their shutil.copy2 -> symlink) ----
out = Path(f"{D}/epvt/bias{a.bias}_{a.split}/ISIC2019_train"); out.mkdir(parents=True, exist_ok=True)
counts = {}
for _, row in df_final.iterrows():
    sub_dir = row[row == 1].index[0]
    label_name = {1: 'mel', 0: 'ben'}.get(row['label'], '!')
    d = out / LETTER.get(sub_dir, sub_dir) / label_name; d.mkdir(parents=True, exist_ok=True)
    t = d / (row['image'] + '.jpg')
    if not t.exists():
        t.symlink_to(ISIC + row['image'] + '.jpg')
    counts[f"{sub_dir}/{label_name}"] = counts.get(f"{sub_dir}/{label_name}", 0) + 1
info = {"train_csv": root_csv, "n_train_csv": len(data), "n_grouped": int(len(df_final)), "counts": dict(sorted(counts.items())),
        "per_domain": {k: int(sum(v for kk, v in counts.items() if kk.startswith(k + '/'))) for k in LETTER}}
for s in ("val", "test"):
    info[f"n_{s}"] = len(pd.read_csv(f"{B}/trap_sets/{s}_bias_{a.bias}_{a.split}.csv"))
(out.parent / "counts.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))
