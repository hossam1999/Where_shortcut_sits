"""A37 row-by-row check (author, 2026-11-19): does DISC's image cache of trap set k belong to trap set k?

DISC's ISICDataset caches the eval-transformed images (Resize(224), ColorJitter(hue=0.2), ToTensor, ImageNet normalisation)
at <root_dir>/features/feature_mat.npy in its own row order: train CSV, then the val and test parts of the test CSV
(train_test_split(test_size=0.8, random_state=0)). A hue shift leaves each pixel's max over R, G, B (V in HSV) unchanged,
so every cached row's V map (un-normalised) is compared with V of a fresh Resize(224) of the image that row's CSV entry
names. A row matches if its mean absolute V difference is < TOL; the cache passes only if every row matches.
A shuffled control (each row against another row's image) must fail, to show the check can tell images apart.

    /workspace/envs/disc/bin/python p8_cache_check.py <k> [--root DIR]     exit 0 = pass (or no cache yet, with --pre)
"""
import argparse, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

TOL = 0.02
MEAN, STD = np.array([0.485, 0.456, 0.406])[:, None, None], np.array([0.229, 0.224, 0.225])[:, None, None]


def rows(k, root):
    S = os.path.join(root, "trap-sets")
    tr = pd.read_csv(f"{S}/isic_annotated_train{k}.csv"); te = pd.read_csv(f"{S}/isic_annotated_test{k}.csv")
    iv, it = train_test_split(np.arange(len(te)), test_size=0.8, random_state=0)
    return np.concatenate([tr.image.values, te.iloc[iv].image.values, te.iloc[it].image.values])


def fresh_v(args):
    root, name = args
    im = Image.open(os.path.join(root, "ISIC2018_Task1-2_Training_Input", name[:-4] + ".jpg")).convert("RGB")
    return (np.asarray(T.Resize((224, 224))(im), np.float32) / 255.0).max(2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("k", type=int)
    ap.add_argument("--root", default=None); ap.add_argument("--pre", action="store_true", help="no cache yet = pass")
    a = ap.parse_args()
    root = a.root or f"{os.environ.get('WTSS_DATA', '/workspace/data')}/disc/isic_set{a.k}"
    f = os.path.join(root, "features", "feature_mat.npy")
    if not os.path.exists(f):
        print(f"set {a.k}: no cache at {f}" + (" (the run builds it from set k's CSVs)" if a.pre else ""))
        sys.exit(0 if a.pre else 2)
    X = np.load(f, mmap_mode="r"); names = rows(a.k, root)
    assert len(X) == len(names), (len(X), len(names))
    with Pool(12) as p:
        V = p.map(fresh_v, [(root, n) for n in names], chunksize=16)
    d = np.empty(len(names)); ctrl = np.empty(len(names)); rng = np.random.default_rng(0); perm = rng.permutation(len(names))
    for i in range(len(names)):
        c = (np.asarray(X[i], np.float64) * STD + MEAN).max(0)
        d[i] = np.abs(c - V[i]).mean(); ctrl[i] = np.abs(c - V[perm[i]]).mean()
    ok = d < TOL
    print(f"set {a.k}: {ok.sum()}/{len(ok)} rows match their own CSV image (mean |dV| {d.mean():.4f}, max {d.max():.4f}); "
          f"shuffled control: {(ctrl < TOL).sum()}/{len(ctrl)} match (mean |dV| {ctrl.mean():.4f}) -> {'PASS' if ok.all() else 'FAIL'}")
    sys.exit(0 if ok.all() else 1)


if __name__ == "__main__":
    main()
