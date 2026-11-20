"""P7 (RR-ClArC, Dreyer et al. AAAI 2024; A28): our arms a, a', b vs our reproduced RR-ClArC, per architecture.

Data: their isic_attacked dataset, split default_rng(0) 80/10/10; training images = the fine-tune data of the architecture
(LSB in p-bias of MEL, A28: 8 / 12.5 / 8 %); artifact label (training only) = LSB inserted yes/no; validation and test scored
clean and attacked (trigger in every image of every class = the paper's "biased" test).
Arms (A28): a  = mask_cmc on their released Vanilla network (penultimate global-average-pooled features: VGG-16 mean of the
                 last conv block (512), ResNet-18 avgpool (512), EfficientNet-B0 avgpool (1,280));
            a' = the same on our reproduced RR-ClArC network of the architecture;
            b  = DINOv2 ViT-B/14, frozen, at 224 px on the unmasked 224-px images (their tensors un-normalised exactly and
                 re-normalised with ImageNet statistics, so the bit-level trigger reaches the encoder), CLS, L2-normalised.
Head: within-class artifact mean difference projected out (only MEL has both groups: one delta, mode pooled),
class-balanced multinomial logistic regression refit on the training set; C on the A21 grid chosen by the mean of clean and
biased validation accuracy (edge reported). Test scored once. "Beaten" (section 2): biased-test accuracy vs our RR-ClArC of
the same architecture, image-only paired bootstrap (10,000), Holm over the 9 P7 comparisons (3 arms x 3 architectures);
VGG-16's comparisons are descriptive (RR-ClArC not reproduced, |delta| 0.039 > 0.03); clean drop > 0.01 flagged.

    cd /workspace/third_party/rrclarc && /workspace/envs/r2r/bin/python <repo>/scripts/replication/p7_arms.py --stage features
    /workspace/venv-wtss/bin/python <repo>/scripts/replication/p7_arms.py --stage heads
Outputs: $WTSS_DATA/p7_arms/<arch>.npz (features) and $WTSS_DATA/results_push/results/replication/rrclarc/p7/arms_* (pushed).
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = os.environ.get("WTSS_DATA", "/workspace/data")
RC = "/workspace/third_party/rrclarc"
W = Path(f"{D}/p7_arms")
RES = Path(f"{D}/results_push/results/replication/rrclarc/p7")
ARCHS = ("resnet18", "efficientnet_b0", "vgg16")
REPRODUCED = {"resnet18": True, "efficientnet_b0": True, "vgg16": False}
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0)


def features():
    import torch, yaml
    sys.path.insert(0, RC)
    from torch.utils.data import DataLoader
    from datasets import get_dataset
    from models import get_fn_model_loader
    from utils.artificial_artifact import get_artifact_kwargs
    torch.hub.set_dir("/root/data/cache/torch_hub")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).eval().cuda()
    IM = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())
    W.mkdir(parents=True, exist_ok=True)
    for arch in ARCHS:
        out = W / f"{arch}.npz"
        if out.exists():
            continue
        cfg = yaml.safe_load(open(f"{RC}/config_files/correcting_isic_attacked/local/{arch}_RRClarc_p7.yaml"))
        ak = get_artifact_kwargs(cfg); mk = get_dataset(cfg["dataset_name"])
        ch = mk(data_paths=cfg["data_paths"], normalize_data=True, p_artifact=cfg["p_artifact"], artifact_type=cfg["artifact_type"],
                attacked_classes=cfg["attacked_classes"], **ak)
        clean = mk(data_paths=cfg["data_paths"], normalize_data=True, attacked_classes=[], **ak)
        att = mk(data_paths=cfg["data_paths"], normalize_data=True, p_artifact=1.0, artifact_type=cfg["artifact_type"],
                 attacked_classes=ch.class_names, **ak)
        sets = {"train": (ch, ch.idxs_train), "val_clean": (clean, ch.idxs_val), "val_attacked": (att, ch.idxs_val),
                "test_clean": (clean, ch.idxs_test), "test_attacked": (att, ch.idxs_test)}
        nets = {"vanilla": f"{D}/rrclarc/checkpoints/checkpoint_{arch}_isic_attacked_lsb.pth",
                "rrclarc": f"{RC}/{cfg['ckpt_path_corrected']}"}
        res = {"class_names": np.array(ch.class_names, dtype="U")}
        art = set(np.asarray(ch.sample_ids_by_artifact["artificial"]).tolist())
        res["a_train"] = np.array([int(i in art) for i in ch.idxs_train])
        models = {}
        for n, ck in nets.items():
            m = get_fn_model_loader(arch)(n_class=len(ch.class_names), ckpt_path=ck).cuda().eval()
            F = []
            mod = m.features if arch == "vgg16" else m.avgpool
            m._p7_hook = mod.register_forward_hook(lambda mm, i, o, F=F: F.append((o.mean((2, 3)) if o.dim() == 4 else o).flatten(1).float().cpu()))
            models[n] = (m, F)
        for s, (ds, idx) in sets.items():
            sub = ds.get_subset_by_idxs(idx); sub.do_augmentation = False   # as their start_model_correction ("Important!")
            dl = DataLoader(sub, batch_size=64, shuffle=False, num_workers=8)
            Xd, ys = [], []
            for n in models:
                models[n][1].clear()
            with torch.no_grad():
                for x, y in dl:
                    x = x.cuda(); ys.append(y)
                    for n, (m, F) in models.items():
                        m(x)
                    z = dino((x * 0.5 + 0.5 - IM[0]) / IM[1])   # their (x - .5) / .5 undone exactly, ImageNet stats for DINOv2
                    Xd.append(torch.nn.functional.normalize(z.float(), dim=1).cpu())
            for n, (m, F) in models.items():
                res[f"X_{n}_{s}"] = torch.cat(F).numpy()
            res[f"X_dino_{s}"] = torch.cat(Xd).numpy(); res[f"y_{s}"] = torch.cat(ys).numpy(); res[f"idx_{s}"] = np.asarray(idx)
            print(arch, s, {k: v.shape for k, v in res.items() if k.endswith(s)}, flush=True)
        for n, (m, F) in models.items():
            m._p7_hook.remove()
        del models; torch.cuda.empty_cache()
        np.savez(out, **res)


def heads():
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8")]
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from wtss.replication_stats import paired_bootstrap, holm, beaten
    rows, preds, sels, comps = [], [], [], []
    for arch in ARCHS:
        z = np.load(W / f"{arch}.npz")
        ref = {v: pd.read_csv(RES / f"{arch}_logits_test_{v}.csv.gz") for v in ("clean", "attacked")}
        for arm, net in (("a_vanilla_mask_cmc", "vanilla"), ("a2_rrclarc_mask_cmc", "rrclarc"), ("b_dino224_mask_cmc", "dino")):
            X = {s: z[f"X_{net}_{s}"] for s in ("train", "val_clean", "val_attacked", "test_clean", "test_attacked")}
            P = CA.Projector(CA.cmc_directions(X["train"], z["y_train"], z["a_train"], "pooled"))
            Zs = {s: P(v) for s, v in X.items()}
            best = None
            for C in C_GRID:
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=0).fit(Zs["train"], z["y_train"])
                acc = {v: float((clf.predict(Zs[f"val_{v}"]) == z[f"y_val_{v}"]).mean()) for v in ("clean", "attacked")}
                sel = (acc["clean"] + acc["attacked"]) / 2
                sels.append({"arch": arch, "arm": arm, "C": C, "val_clean_acc": acc["clean"], "val_biased_acc": acc["attacked"], "val_mean": sel,
                             "converged": bool(clf.n_iter_.max() < 5000)})
                if best is None or sel > best[1]:
                    best = (clf, sel, C)
            clf, sel, C = best
            r = {"arch": arch, "arm": arm, "C": C, "C_edge": "lowest" if C == C_GRID[0] else "highest" if C == C_GRID[-1] else "", "val_mean_acc": sel}
            for v in ("clean", "attacked"):
                y = z[f"y_test_{v}"]; p = clf.predict_proba(Zs[f"test_{v}"])
                r[f"test_{v}_acc"] = float((p.argmax(1) == y).mean()); r[f"test_{v}_f1"] = float(f1_score(y, p.argmax(1), average="macro"))
                preds.append(pd.DataFrame(p, columns=[f"p_{c}" for c in z["class_names"]]).assign(idx=z[f"idx_test_{v}"], y=y, arch=arch, arm=arm, variant=v))
                R = ref[v].set_index("idx").loc[z[f"idx_test_{v}"]]
                assert (R.y.to_numpy() == y).all()
                rl = R[[c for c in R.columns if c.startswith("logit_")]].to_numpy()
                r[f"rrclarc_test_{v}_acc"] = float((rl.argmax(1) == y).mean())
                if v == "attacked":
                    b = paired_bootstrap({0: (z[f"idx_test_{v}"].astype(str).tolist(), y, p, rl)}, "accuracy")
                    comps.append({"arch": arch, "arm": arm, "reproduced": REPRODUCED[arch], **{k: b[k] for k in ("ours_mean", "theirs_mean", "delta", "ci95_lo", "ci95_hi", "p_boot_two_sided")}})
            r["clean_drop_vs_rrclarc"] = r["test_clean_acc"] - r["rrclarc_test_clean_acc"]
            rows.append(r)
    c = pd.DataFrame(comps); c["p_holm"] = holm(c.p_boot_two_sided.tolist())
    c["meets_beaten_rule"] = [bool(beaten(x, x.p_holm)) for _, x in c.iterrows()]
    c["verdict"] = np.where(c.reproduced, np.where(c.meets_beaten_rule, "beaten", "not beaten"), "descriptive (RR-ClArC not reproduced)")
    r = pd.DataFrame(rows); RES.mkdir(parents=True, exist_ok=True)
    r.to_csv(RES / "arms_summary.csv", index=False); c.to_csv(RES / "arms_comparison.csv", index=False)
    pd.concat(preds).to_csv(RES / "arms_test_probs.csv.gz", index=False); pd.DataFrame(sels).to_csv(RES / "arms_selection.csv", index=False)
    m = c.merge(r, on=["arch", "arm"])
    md = ["# P7 (RR-ClArC, A28): our arms vs our reproduced RR-ClArC — machine 2", "",
          "Biased test = LSB trigger in every image of every class (Table 1 'biased'); accuracy; image-only paired bootstrap "
          "(10,000), Holm over the 9 comparisons; 'beaten' = higher accuracy, 95 % CI above 0, Holm p < 0.05 (section 2); "
          "VGG-16 descriptive (RR-ClArC not reproduced). Clean drop vs RR-ClArC > 0.01 flagged.", "",
          "| arch | arm | C (edge) | biased acc | RR-ClArC biased | Δ [95 % CI] | p Holm | clean acc (Δ vs RR-ClArC) | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, x in m.iterrows():
        flag = " ⚠" if x.clean_drop_vs_rrclarc < -0.01 else ""
        md.append(f"| {x.arch} | {x.arm} | {x.C:g} {('(' + x.C_edge + ')') if x.C_edge else ''} | {x.test_attacked_acc:.3f} | {x.theirs_mean:.3f} | "
                  f"{x.delta:+.3f} [{x.ci95_lo:+.3f}, {x.ci95_hi:+.3f}] | {x.p_holm:.4f} | {x.test_clean_acc:.3f} ({x.clean_drop_vs_rrclarc:+.3f}){flag} | {x.verdict} |")
    (RES / "arms_comparison.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["features", "heads"], required=True)
    {"features": features, "heads": heads}[ap.parse_args().stage]()
