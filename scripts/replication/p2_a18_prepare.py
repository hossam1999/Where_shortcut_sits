"""A18 (P2 RR-ClArC at the paper's stated settings): a weights-only copy of our Vanilla checkpoint (so their
get_optimizer builds a fresh Adam at the configured lr instead of loading the decayed 1e-5 state) and the 3 configs
(λ = 5e9, lr 1e-4, layers identity_1 / identity_2 / last_conv, 10 epochs; everything else from their configs).

    python scripts/replication/p2_a18_prepare.py      # (pahde env)
"""
import os
from pathlib import Path

import torch
import yaml

D = os.environ.get("WTSS_DATA", "/workspace/data")
R = Path(f"{D}/replication/pahde")
SRC = R / "runs/vanilla_resnet50d_seed0/checkpoints/config/checkpoint_resnet50d_last.pth"
W = R / "runs/vanilla_resnet50d_seed0/vanilla_weights_only.pth"
OUTD = R / "mitigation_a18"

c = torch.load(SRC, map_location="cpu", weights_only=False)
torch.save({"epoch": c["epoch"], "model_state_dict": c["model_state_dict"]}, W)
OUTD.mkdir(exist_ok=True)
for layer in ("identity_1", "identity_2", "last_conv"):
    cfg = yaml.safe_load(open(R / f"mitigation/resnet50d_RRClarc_lamb10_adam_lr0.001_{layer}.yaml"))
    cfg.update(lamb=5e9, lr=1e-4, ckpt_path=str(W), checkpoint_dir_corrected=str(R / "checkpoints_corrected_a18"))
    assert cfg["num_epochs"] == 10 and cfg["layer_name"] == layer
    yaml.safe_dump(cfg, open(OUTD / f"resnet50d_RRClarc_lamb5e9_adam_lr0.0001_{layer}.yaml", "w"))
print("weights-only:", W, "keys", list(torch.load(W, map_location="cpu", weights_only=False).keys()))
print(sorted(p.name for p in OUTD.iterdir()))
