"""A19 arm a': penultimate features of the selected A18 RR-ClArC network (their best model), for mask_cmc on top of it.
Two input modes; A19 fixes which one is used (written to $WTSS_DATA/replication/pahde/A19_INPUT when A19 is recorded):
  unmasked — as arm a: their dataset images as the network was trained (p2_score.py --features);
  lesion   — the same rendered images (p2_render.py) with the lesion ROI (p2_lesion_rois.py) applied as in arm b
             (outside replaced by the mean colour, wtss.ops.apply_roi_mask), then their normalisation.

    python scripts/replication/p2_aprime_features.py --config <chosen A18 yaml> --ckpt <its last.ckpt> --input unmasked|lesion
Writes $WTSS_DATA/replication/pahde/scores_a18/aprime_features_<input>.npz (train_as_trained, val_*, test_*).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = os.environ.get("WTSS_DATA", "/workspace/data")
R = Path(f"{D}/replication/pahde")
SETS = ("train_as_trained", "val_clean", "val_biased", "test_clean", "test_biased")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", choices=["unmasked", "lesion"], required=True)
    a = ap.parse_args()
    out = R / f"scores_a18/aprime_features_{a.input}.npz"
    if a.input == "unmasked":   # exactly arm a's extraction, on the RR-ClArC network
        tmp = R / "scores_a18/aprime_unmasked_scores.csv.gz"
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/replication/p2_score.py"), "--config", a.config, "--ckpt", a.ckpt,
                        "--out", str(tmp), "--features"], check=True)
        Path(str(tmp).replace(".csv.gz", "_features.npz")).replace(out)
        return
    import torch
    import yaml
    sys.path[:0] = ["/workspace/third_party/pahde", str(ROOT / "src")]
    from models import get_fn_model_loader
    from PIL import Image
    from wtss.ops import apply_roi_mask
    cfg = yaml.safe_load(open(a.config))
    model = get_fn_model_loader(model_name=cfg["model_name"])(n_class=8, ckpt_path=a.ckpt, device="cuda").cuda().eval()
    feats = {}
    model.fc.register_forward_hook(lambda m, i, o: feats.__setitem__("f", i[0].detach().cpu()))
    z = np.load(R / "lesion_roi224.npz")
    roi = dict(zip(z["ids"].astype(str), np.unpackbits(z["bits"], axis=-1)[..., :224].astype(bool)))
    import pandas as pd
    F = {}
    for k in SETS:
        part = pd.read_csv(R / f"render224/{k}.csv")
        ims = np.load(R / f"render224/{k}.npy", mmap_mode="r")
        fs = []
        with torch.no_grad():
            for s0 in range(0, len(part), 64):
                x = np.stack([np.asarray(apply_roi_mask(Image.fromarray(np.asarray(ims[j])), roi[str(part.image[j])]))
                              for j in range(s0, min(s0 + 64, len(part)))])
                t = (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5   # their T.Normalize([0.5]*3, [0.5]*3)
                model(t.cuda()); fs.append(feats["f"])
        F[k] = torch.cat(fs).numpy()
        print(k, F[k].shape, flush=True)
    np.savez_compressed(out, **F)


if __name__ == "__main__":
    main()
