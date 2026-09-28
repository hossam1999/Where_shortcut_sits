"""A1 — agreement of our labels with each independent source.

  python scripts/round6/label_agreement.py --cohort isic|thyroid|capsule|ovary
  python scripts/round6/label_agreement.py --cohort isic --smoke
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

S = 518


def _resize_mask(arr, size=S) -> np.ndarray:
    im = Image.fromarray(arr.astype(np.uint8) * 255)
    return np.asarray(im.resize((size, size), Image.NEAREST)) >= 128


def _agree_block(y, a, b, kind="kappa"):
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b, float)

    def pack(mask):
        if mask.sum() == 0:
            return {"n": 0}
        aa, bb, yy = a[mask], b[mask], y[mask]

        def stat(idx):
            if kind == "kappa":
                return C.kappa_binary(aa[idx].astype(int), np.rint(bb[idx]).astype(int))
            return float(np.nanmean(bb[idx]))

        return C.boot_ci(stat, len(aa))

    out = {"all": pack(np.ones(len(y), bool))}
    for lab in (0, 1):
        out[f"y{lab}"] = pack(y == lab)
    return out


def _ima_rows(ids, cache, idmap):
    meta = pd.read_csv(C.EXT / "ima_seg_metadata.csv")
    meta["image_id"] = meta.ISIC_id.map(C.isic_key)
    meta = meta[meta.image_id.isin(set(ids))]
    zpath = C.EXT / "segs.zip"
    with zipfile.ZipFile(zpath) as z:
        names = {Path(n).name: n for n in z.namelist()}
        rows = []
        for image_id, g in meta.groupby("image_id"):
            masks = []
            for fn in g.seg_filename:
                key = Path(str(fn)).name
                if key not in names:
                    continue
                raw = z.read(names[key])
                m = np.asarray(Image.open(io.BytesIO(raw)).convert("L")) >= 128
                masks.append(_resize_mask(m))
            if not masks:
                continue
            maj = (np.mean(masks, axis=0) >= 0.5)
            raw_id = idmap.get(str(image_id), str(image_id))
            if raw_id not in cache.index:
                continue
            rgb, roi, art = cache.get(raw_id)
            roi_b = np.asarray(roi) > 0
            hair = np.asarray(art) > 0
            if roi_b.shape != maj.shape:
                roi_b = _resize_mask(roi_b)
                hair = _resize_mask(hair)
            dice, iou = C.dice_iou(roi_b, maj)
            hsum = int(hair.sum())
            r_ima = float((hair & maj).sum() / hsum) if hsum else float("nan")
            rows.append({"image_id": image_id, "dice_ima": dice, "iou_ima": iou, "r_ima": r_ima, "n_annotators": len(masks)})
            if len(rows) % 250 == 0:
                C.log(ima=len(rows))
    return pd.DataFrame(rows)


def _kabir_rows(kabir, cache, idmap):
    rows = []
    for r in kabir.itertuples(index=False):
        raw_id = idmap.get(str(r.image_id), str(r.image_id))
        if raw_id not in cache.index:
            continue
        km = np.asarray(Image.open(r.mask_path).convert("L")) >= 128
        native_px = int(km.sum())
        rgb, roi, art = cache.get(raw_id)
        k518 = _resize_mask(km)
        hair = np.asarray(art) > 0
        roi_b = np.asarray(roi) > 0
        dice, iou = C.dice_iou(hair, k518)
        hsum = int(k518.sum())
        r_k = float((k518 & roi_b).sum() / hsum) if hsum else float("nan")
        rows.append({"image_id": r.image_id, "kabir_px": native_px, "kabir_present": int(native_px > 30),
                     "dice_kabir": dice, "iou_kabir": iou, "r_kabir": r_k})
    return pd.DataFrame(rows)


def _capsule_rows(matched, cache):
    rows = []
    for r in matched.itertuples(index=False):
        m = np.asarray(Image.open(r.mask_path).convert("L"))
        expert = _resize_mask(m < 128)  # black = contaminated
        rgb, roi, art = cache.get(str(r.image_id))
        ours = np.asarray(art) > 0
        roi_b = np.asarray(roi) > 0
        dice, iou = C.dice_iou(ours, expert)
        frac = float(expert.mean())
        esum = int(expert.sum())
        r_ex = float((expert & roi_b).sum() / esum) if esum else float("nan")
        rows.append({"image_id": r.image_id, "iou_expert": iou, "dice_expert": dice,
                     "contam_frac_expert": frac, "r_expert": r_ex})
    return pd.DataFrame(rows)


def _cell_from_r(present, r):
    return C.source_cell(bool(present), None if pd.isna(r) else float(r))


def run_cohort(name: str, smoke: bool):
    out = C.out_root(smoke) / "agreement" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "agreement.json").exists() and not smoke:
        C.log(cohort=name, status="cached")
        return
    root = C.out_root(smoke)
    info = C.cohort_frame(name)
    c = info["c"]
    if name == "isic":
        c = c.copy()
        c["kid"] = c.image_id.map(C.isic_key)
    else:
        c = c.copy()
        c["kid"] = c.image_id.astype(str)
    per = c[["image_id", "kid", "y", "cell", "present", "free", "r_our"]].copy()
    report = {"cohort": name}

    if name == "isic":
        derm_p = root / "matches" / "isic_derm.csv"
        if derm_p.exists():
            derm = pd.read_csv(derm_p)
            if len(derm):
                derm["kid"] = derm.image_id.map(C.isic_key)
                per = per.merge(derm[["kid", "hair_present"]], on="kid", how="left")
                m = per.hair_present.notna()
                report["derm_kappa"] = _agree_block(per.y[m], per.present[m].astype(int), per.hair_present[m], "kappa")
                per["cell_derm"] = [C.source_cell(int(v), None) if pd.notna(v) else "missing" for v in per.hair_present]
        ima_p = root / "matches" / "isic_ima.csv"
        if ima_p.exists() and (C.EXT / "segs.zip").exists():
            ima = pd.read_csv(ima_p)
            ima = ima[~ima.excluded_isic2018_train.astype(bool)]
            ids = set(ima.image_id.map(C.isic_key))
            if smoke:
                ids = set(list(ids)[:8])
            idmap = dict(zip(per.kid.astype(str), per.image_id.astype(str)))
            pix = _ima_rows(ids, info["cache"], idmap)
            if len(pix):
                pix["kid"] = pix.image_id.map(C.isic_key)
                per = per.merge(pix.drop(columns=["image_id"]), on="kid", how="left")
                m = per.dice_ima.notna()
                report["ima_dice"] = _agree_block(per.y[m], per.present[m], per.dice_ima[m], "mean")
                report["ima_iou"] = _agree_block(per.y[m], per.present[m], per.iou_ima[m], "mean")
                both = m & per.present.astype(bool) & per.r_ima.notna() & per.r_our.notna()
                our_t = np.array([_cell_from_r(True, r) for r in per.r_our])
                ima_t = np.array([_cell_from_r(True, r) for r in per.r_ima])
                agree = (our_t == ima_t).astype(float)
                report["ima_trap_class"] = _agree_block(per.y[both], per.present[both], agree[both], "mean")
                per["cell_ima"] = ["missing" if pd.isna(r) else _cell_from_r(bool(p), r) for p, r in zip(per.present, per.r_ima)]
        kab_p = root / "matches" / "isic_kabir.csv"
        if kab_p.exists():
            kab = pd.read_csv(kab_p)
            if smoke and len(kab):
                kab = kab.head(8)
            if len(kab):
                idmap = dict(zip(per.kid.astype(str), per.image_id.astype(str)))
                pix = _kabir_rows(kab, info["cache"], idmap)
                pix["kid"] = pix.image_id.map(C.isic_key)
                per = per.merge(pix.drop(columns=["image_id"]), on="kid", how="left")
                m = per.dice_kabir.notna()
                report["kabir_dice"] = _agree_block(per.y[m], per.present[m], per.dice_kabir[m], "mean")
                report["kabir_kappa"] = _agree_block(per.y[m], per.present[m].astype(int), per.kabir_present[m], "kappa")
                per["cell_kabir"] = [C.source_cell(int(p), None if pd.isna(r) else float(r)) if pd.notna(p) else "missing"
                                     for p, r in zip(per.kabir_present, per.r_kabir)] if "kabir_present" in per else "missing"
    if name == "capsule":
        mp = root / "matches" / "capsule_expert.csv"
        if mp.exists():
            matched = pd.read_csv(mp)
            if smoke and len(matched):
                matched = matched.head(8)
            if len(matched):
                pix = _capsule_rows(matched, info["cache"])
                per = per.merge(pix, on="image_id", how="left")
                m = per.iou_expert.notna()
                report["capsule_iou"] = _agree_block(per.y[m], per.present[m], per.iou_expert[m], "mean")
                per["cell_expert"] = [C.source_cell(fr >= 0.10, None if pd.isna(r) else float(r)) if pd.notna(fr) else "missing"
                                      for fr, r in zip(per.contam_frac_expert, per.r_expert)]
                # artifact-free is frac < 0.03; the gap 0.03-0.10 is not present
                cells = []
                for fr, r in zip(per.contam_frac_expert, per.r_expert):
                    if pd.isna(fr):
                        cells.append("missing")
                    elif fr < 0.03:
                        cells.append("artifact_free")
                    elif fr >= 0.10:
                        cells.append(C.source_cell(True, None if pd.isna(r) else float(r)))
                    else:
                        cells.append("gap")
                per["cell_expert"] = cells
    if name in ("thyroid", "ovary"):
        bp = root / "matches" / f"{name}_busclean.csv"
        if bp.exists():
            b = pd.read_csv(bp)
            per = per.merge(b.drop(columns=["cohort"], errors="ignore"), on="image_id", how="left")
            # presence kappa on the registered contrast: marker_px == 0 vs >= 15 (drop the gap)
            m = per.bus_present.notna() & per.cell.isin(["artifact_free", "trapA", "trapB", "mid"])
            # mid is present; gap excluded. artifact_free is absent.
            m = per.bus_present.notna() & (per.free | per.present)
            report["busclean_kappa"] = _agree_block(per.y[m], per.present[m].astype(int), per.bus_present[m], "kappa")
            side = per.present & per.cell.isin(["trapA", "trapB"]) & per.bus_cell.isin(["trapA", "trapB"])
            agree = (per.cell == per.bus_cell).astype(float)
            report["busclean_location"] = _agree_block(per.y[side], per.present[side], agree[side], "mean")
            report["informative"] = {"BUSClean": "not_evaluable",
                                     "reason": "MedGemma did not run, so the two-other-labels agreement set is empty. "
                                               "A model labeller is uninformative unless kappa >= 0.40 on that set."}
            per["cell_bus"] = per.bus_cell if "bus_cell" in per else "missing"

    # contradiction flags for A2, only from sources whose coverage gate is analysed
    gates = {}
    cov_p = root / "coverage.csv"
    if cov_p.exists():
        cov = pd.read_csv(cov_p)
        sub = cov[(cov.cohort == name) & (cov.cell == "all")]
        gates = dict(zip(sub.source, sub.gate))
    src_cols = [col for col in per.columns if col.startswith("cell_")]
    contrad = np.zeros(len(per), bool)
    verified = np.zeros(len(per), bool)
    informative_cols = []
    for col in src_cols:
        gate_name = {"cell_derm": "DermArtifactDB", "cell_ima": "IMA++", "cell_kabir": "Kabir",
                     "cell_expert": "Mendeley_vmxhn95j8z", "cell_bus": "BUSClean"}.get(col)
        if gates.get(gate_name) != "analysed":
            continue
        if col == "cell_bus":
            continue  # not evaluable without the third label
        informative_cols.append(col)
        src = per[col].astype(str)
        bad = np.array([C.contradicts(o, s) for o, s in zip(per.cell, src)])
        known = src.ne("missing") & src.ne("nan")
        contrad |= bad & known
        verified |= known & ~bad
    verified &= ~contrad
    per["contradicted"] = contrad
    per["verified"] = verified
    per["unverified"] = ~contrad & ~verified
    report["informative_sources"] = informative_cols
    report["gates"] = gates
    per.to_csv(out / "per_image.csv", index=False)
    (out / "agreement.json").write_text(json.dumps(report, indent=2))
    C.log(cohort=name, contradicted=int(contrad.sum()), verified=int(verified.sum()), unverified=int((~contrad & ~verified).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=C.COHORTS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run_cohort(a.cohort, a.smoke)


if __name__ == "__main__":
    main()
