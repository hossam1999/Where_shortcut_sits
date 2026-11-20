"""P7 (RR-ClArC, Dreyer et al. AAAI 2024; A28): write the three RR-ClArC correction configs in their format, settings as A28
(the paper wins over the code): fine-tune 10 epochs from the released Vanilla checkpoint, cross-entropy, lr 1e-4, SGD
(VGG-16, ResNet-18) / Adam (EfficientNet-B0), CAV layer features.28 / last_conv / last_conv, `all_logits_random`, `l2_mean`,
`cavs_max` (their start_model_correction), `signal`, CAV on the training set; lambda from their Table A.3 (VGG-16 5e8,
ResNet-18 1e11, EfficientNet-B0 5e9); artifact rate of the fine-tune data = the paper's p-bias (Table A.1) 8.0 / 12.5 / 8.0 %
of MEL; batch 64 (their local config); LSB trigger and lsb_factor exactly as their generator. W&B offline (their trainer
requires a logger). Configs: /workspace/third_party/rrclarc/config_files/correcting_isic_attacked/local/<arch>_RRClarc_p7.yaml

    /workspace/envs/r2r/bin/python scripts/replication/p7_configs.py
"""
import importlib.util, os, yaml

RC = "/workspace/third_party/rrclarc"
D = os.environ.get("WTSS_DATA", "/workspace/data")
ARCH = {  # model: (layer, optimizer, lambda, p_artifact)
    "vgg16": ("features.28", "sgd", 5e8, 0.08),
    "resnet18": ("last_conv", "sgd", 1e11, 0.125),
    "efficientnet_b0": ("last_conv", "adam", 5e9, 0.08),
}


def lsb_trigger():
    """Their generator's trigger string, read from the generator source without running it."""
    src = open(f"{RC}/config_files/config_generator_correcting_isic_attacked.py").read()
    start = src.index("lsb_trigger = "); end = src.index("\ndef store_local")
    ns = {}; exec(src[start:end], ns)
    return ns["lsb_trigger"]


def main():
    trig = lsb_trigger(); out = f"{RC}/config_files/correcting_isic_attacked/local"; os.makedirs(out, exist_ok=True)
    for m, (layer, opt, lamb, p) in ARCH.items():
        name = f"{m}_RRClarc_p7"
        cfg = {"num_epochs": 10, "device": "cuda", "dataset_name": "isic_attacked", "loss": "cross_entropy", "img_size": 224,
               "attacked_classes": ["MEL"], "p_artifact": p, "model_name": m, "layer_name": layer, "artifact_type": "lsb",
               "lsb_factor": 2 if m == "efficientnet_b0" else 3, "lsb_trigger": trig, "artifact": "artificial", "lr": 1e-4,
               "optimizer": opt, "method": "RRClarc", "lamb": lamb, "direction_mode": "signal", "compute": "l2_mean",
               "criterion": "all_logits_random", "ckpt_path": f"{D}/rrclarc/checkpoints/checkpoint_{m}_isic_attacked_lsb.pth",
               "ckpt_path_corrected": f"checkpoints/{name}/last.ckpt", "data_paths": [f"{D}/r2r/dataset_isic2019"],
               "batch_size": 64, "wandb_project_name": "p7_rrclarc", "wandb_api_key": "offline", "unique_wandb_ids": True}
        yaml.dump(cfg, open(f"{out}/{name}.yaml", "w"), default_flow_style=False)
        print(f"{out}/{name}.yaml", {k: cfg[k] for k in ("layer_name", "optimizer", "lamb", "p_artifact", "lr")})


if __name__ == "__main__":
    main()
