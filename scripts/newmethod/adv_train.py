"""Exploratory candidate (b) (docs/PREREGISTRATION_NEWMETHOD.md, DRAFT): class-conditional adversarial head with gradient
reversal, on top of arm d's recipe (EfficientNet-B0, ImageNet weights, ROI-masked inputs, Adam 1e-4, batch 64, <= 50
epochs, early stopping patience 5 on the setting's validation metric, seed 1234, mixed precision). VALIDATION ONLY.

Development setting: P1 location test, tube inside (armd_train.P1LocInside: arm b's rendering at 128 px; validation
reverses the tube-label link). No test environment is read.

    L = CE(h(z), y) + BCE(g(GRL_alpha(z))[y], A)
z = pooled 1,280-d features; h = disease head; g = artifact head with one logit per class (the true class's logit is used:
artifact presence predicted within each class); GRL = identity forward, -alpha x gradient backward. After training,
mask_cmc (pooled / per_class) + class-balanced logistic head on the features, C in the A21 grid; reported: validation
AUROC (reversed) and a within-class tube probe AUROC, beside arm d (alpha = 0, same encoder recipe).

    python scripts/newmethod/adv_train.py --alpha 1 --stage train|features|heads
Outputs: $WTSS_DATA/newmethod/adv_p1loc_inside_a<alpha>/ and results/newmethod/adv_p1loc_inside.csv (validation only).
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication"), str(ROOT / "scripts" / "newmethod")]
import armd_train as A  # noqa: E402

D = os.environ.get("WTSS_DATA", "/workspace/data")


class GRL:
    @staticmethod
    def apply(x, alpha):
        import torch

        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, t):
                return t.view_as(t)

            @staticmethod
            def backward(ctx, g):
                return -alpha * g
        return _F.apply(x)


class TubeData:
    """(image, label, tube) from the P1 location adapter's cached, masked images."""

    def __init__(self, S, k):
        self.base = S.dataset(k, False); self.a = S.labels(k)[1]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, j):
        import torch
        x, y = self.base[j]
        return x, y, torch.tensor(float(self.a[j]))


def train(S, out, alpha, workers=8, max_batches=None, max_epochs=A.MAX_EPOCHS, pretrained=True):
    import time
    import torch
    torch.manual_seed(A.SEED); np.random.seed(A.SEED)
    dev = A.dev()
    model = A.make_model(S.n_out, pretrained).to(dev)
    adv = torch.nn.Linear(model.classifier[1].in_features, S.n_out).to(dev)   # one artifact logit per class
    feats = {}
    model.classifier.register_forward_hook(lambda m, i, o: feats.__setitem__("z", i[0]))
    opt = torch.optim.Adam(list(model.parameters()) + list(adv.parameters()), lr=A.LR)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    es, log, start = A.EarlyStop(), [], 0
    if (out / "checkpoint.pt").exists():
        c = torch.load(out / "checkpoint.pt", weights_only=False)
        model.load_state_dict(c["model"]); adv.load_state_dict(c["adv"]); opt.load_state_dict(c["opt"]); scaler.load_state_dict(c["scaler"])
        es, log, start = A.EarlyStop(best=c["best"], bad=c["bad"]), c["log"], c["epoch"] + 1
    g = torch.Generator(); g.manual_seed(A.SEED)
    tr = torch.utils.data.DataLoader(TubeData(S, S.train_set), batch_size=A.BS, shuffle=True, drop_last=True, num_workers=workers,
                                     worker_init_fn=A._one_thread, generator=g)
    va = {k: A.loader(S.dataset(k, False), False, workers) for k in S.val_sets}
    ys = {k: S.labels(k)[0] for k in S.val_sets}
    for ep in range(start, max_epochs):
        if es.stop:
            break
        model.train(); t0, tl, al = time.time(), [], []
        for b, (x, y, a) in enumerate(tr):
            if max_batches and b >= max_batches:
                break
            x, y, a = x.to(dev), y.to(dev), a.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x)
                z = feats["z"]
                ga = adv(GRL.apply(z.float(), alpha)).gather(1, y[:, None])[:, 0]
                lc = torch.nn.functional.cross_entropy(logits, y)
                la = torch.nn.functional.binary_cross_entropy_with_logits(ga.float(), a)
                loss = lc + la
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl.append(float(lc)); al.append(float(la))
        preds = {k: (ys[k], A.predict(model, dl)[0]) for k, dl in va.items()}
        v = S.val_metric(preds); improved = es.step(v)
        log.append({"epoch": ep, "ce": float(np.mean(tl)), "adv_bce": float(np.mean(al)), "val_metric": v, "minutes": (time.time() - t0) / 60,
                    "best": es.best, "bad": es.bad})
        pd.DataFrame(log).to_csv(out / "log.csv", index=False); print(log[-1], flush=True)
        if improved:
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "adv": adv.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": es.best, "bad": es.bad, "log": log}, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(out / "checkpoint.pt")
    (out / "TRAIN_DONE").write_text("ok")


def features(S, out, workers=8):
    import torch
    model = A.make_model(S.n_out).to(A.dev()); model.load_state_dict(torch.load(out / "best.pt", map_location=A.dev())); model.eval()
    F = {}
    for k in ("train", "valid"):   # validation only: no test environment is read
        F[k] = A.predict(model, A.loader(S.dataset(k, False), False, workers), feats=True)[1]
    np.savez_compressed(out / "features_trainval.npz", **F)


def heads(S, out, alpha):
    import candidates as CA
    from dev_ccleace import head, within_class_probe
    F = np.load(out / "features_trainval.npz")
    y = {k: S.labels(k)[0] for k in ("train", "valid")}; a = {k: S.labels(k)[1] for k in ("train", "valid")}
    rows = []
    for name, T in (("none", lambda Z: Z), ("mask_cmc pooled", CA.Projector(CA.cmc_directions(F["train"], y["train"], a["train"], "pooled")))):
        Ztr, Zv = T(F["train"]), T(F["valid"])
        v, C = head(Ztr, y["train"], Zv, y["valid"])
        rows.append({"alpha": alpha, "post_hoc": name, "val_auroc_reversed": round(v, 3), "C": C,
                     "tube_probe_val": round(within_class_probe(Ztr, y["train"], a["train"], Zv, y["valid"], a["valid"]), 3),
                     "epochs": len(pd.read_csv(out / "log.csv"))})
        print(rows[-1], flush=True)
    f = ROOT / "results/newmethod/adv_p1loc_inside.csv"; f.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    pd.concat([old[old.alpha != alpha] if len(old) else old, pd.DataFrame(rows)]).to_csv(f, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--stage", choices=["train", "features", "heads"], required=True)
    a = ap.parse_args()
    S = A.P1LocInside()
    out = Path(f"{D}/newmethod/adv_p1loc_inside_a{a.alpha:g}"); out.mkdir(parents=True, exist_ok=True)
    {"train": lambda: train(S, out, a.alpha), "features": lambda: features(S, out), "heads": lambda: heads(S, out, a.alpha)}[a.stage]()
