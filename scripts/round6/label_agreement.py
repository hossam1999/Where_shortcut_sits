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
            r_our518 = float((hair & roi_b).sum() / hsum) if hsum else float("nan")
            rows.append({"image_id": image_id, "dice_ima": dice, "iou_ima": iou, "r_ima": r_ima, "r_our518": r_our518,
                         "n_annotators": len(masks)})
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
        osum = int(hair.sum())
        r_o = float((hair & roi_b).sum() / osum) if osum else float("nan")
        rows.append({"image_id": r.image_id, "kabir_px": native_px, "kabir_present": int(native_px > 30),
                     "dice_kabir": dice, "iou_kabir": iou, "r_kabir": r_k, "r_our518_k": r_o})
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
    keep = ["image_id", "kid", "y", "cell", "present", "free", "r_our"] + [k for k in ("source", "mask_src") if k in c.columns]
    per = c[keep].copy()
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
                # like for like: both overlaps at 518 px with our hair mask; our ROI vs the IMA++ ROI
                both = m & per.present.astype(bool) & per.r_ima.notna() & per.r_our518.notna()
                our_t = np.array([_cell_from_r(True, r) if pd.notna(r) else "missing" for r in per.r_our518])
                ima_t = np.array([_cell_from_r(True, r) if pd.notna(r) else "missing" for r in per.r_ima])
                agree = (our_t == ima_t).astype(float)
                report["ima_trap_class"] = _agree_block(per.y[both], per.present[both], agree[both], "mean")
                if "source" in per:
                    report["ima_dice_by_roi_source"] = {
                        s: _agree_block(per.y[m & (per.source == s)], per.present[m & (per.source == s)],
                                        per.dice_ima[m & (per.source == s)], "mean")["all"] for s in sorted(per.source[m].unique())}
                # a source cell is used only where our 518-px class matches our registered cell (else resolution-ambiguous)
                per["cell_ima"] = [("missing" if (pd.isna(ri) or pd.isna(ro) or not p) else
                                    (_cell_from_r(True, ri) if _cell_from_r(True, ro) == cell else "missing"))
                                   for p, ri, ro, cell in zip(per.present, per.r_ima, per.r_our518, per.cell)]
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
                cells = []
                for p_, rk, ro, cell in zip(per.kabir_present, per.r_kabir, per.r_our518_k, per.cell):
                    if pd.isna(p_):
                        cells.append("missing")
                    elif not int(p_) or pd.isna(rk) or pd.isna(ro):
                        cells.append(C.source_cell(int(p_), None))
                    else:
                        cells.append(C.source_cell(1, float(rk)) if _cell_from_r(True, ro) == cell else "present")
                per["cell_kabir"] = cells
    if name == "capsule":
        mp = root / "matches" / "capsule_expert.csv"
        if mp.exists():
            matched = pd.read_csv(mp)
            if "mask_src" in c.columns:
                expert_ids = set(c.loc[c.mask_src.astype(str) == "expert", "image_id"].astype(str))
                report["capsule_excluded_expert_labelled"] = int(matched.image_id.astype(str).isin(expert_ids).sum())
                matched = matched[~matched.image_id.astype(str).isin(expert_ids)]
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
            per["cell_bus"] = per.bus_cell.fillna("missing") if "bus_cell" in per else "missing"
            free_rate = float(per.loc[per.free & per.bus_present.notna(), "bus_present"].mean())
            report["busclean_positive_rate_on_caliper_free"] = free_rate
        mgp = root / "matches" / f"{name}_medgemma.csv"
        if mgp.exists():
            g = pd.read_csv(mgp)
            g["image_id"] = g.image_id.astype(str)
            per = per.merge(g, on="image_id", how="left")
            pres = per.mg_presence.map({"yes": 1, "no": 0})
            per["mg_present"] = pres
            m = pres.notna() & (per.free | per.present)
            report["medgemma_kappa"] = _agree_block(per.y[m], per.present[m].astype(int), pres[m], "kappa")
            per["cell_mg"] = ["missing" if pd.isna(pp) else ("artifact_free" if pp == 0 else
                              ("trapA" if lo == "yes" else "trapB" if lo == "no" else "present"))
                              for pp, lo in zip(pres, per.mg_location)]
            side = per.present & per.cell.isin(["trapA", "trapB"]) & per.cell_mg.isin(["trapA", "trapB"])
            report["medgemma_location"] = _agree_block(per.y[side], per.present[side], (per.cell == per.cell_mg).astype(float)[side], "mean")
        # informativeness (A1): kappa >= 0.40 against the images on which the two other caliper labels agree
        inf = {}
        ours = pd.Series(np.where(per.free, 0, np.where(per.present, 1, np.nan)), index=per.index)
        bus = per.bus_present if "bus_present" in per else pd.Series(np.nan, index=per.index)
        mgv = per.mg_present if "mg_present" in per else pd.Series(np.nan, index=per.index)
        bus_valid = "bus_present" in per and report.get("busclean_positive_rate_on_caliper_free", 1.0) <= 0.5
        if "bus_present" in per and not bus_valid:
            inf["BUSClean"] = {"status": "invalid", "reason": "positive on more than half of the caliper-free images: "
                               "it detects on-screen annotation, not calipers (validity guard, Amendment 2)"}
        for lab, x, other in (("BUSClean", bus, mgv), ("MedGemma", mgv, bus)):
            if lab in inf or x.notna().sum() == 0:
                continue
            other_ok = other.notna().sum() > 0 and (lab == "BUSClean" or bus_valid)
            if not other_ok:
                inf[lab] = {"status": "pending_human", "reason": "only one model labeller is available; judged against "
                            "the blinded rating (A4/A4b) with the same threshold (Amendment 2)"}
                continue
            sel = ours.notna() & other.notna() & x.notna() & (ours == other)
            k = C.kappa_binary(ours[sel].astype(int), x[sel].astype(int)) if sel.sum() else float("nan")
            inf[lab] = {"status": "informative" if np.isfinite(k) and k >= 0.40 else "uninformative", "kappa": k,
                        "n": int(sel.sum())}
        fb = root / "rating" / "medgemma_informativeness.json"
        if fb.exists():  # Amendment 2: verdict from the blinded rating once it exists
            human = json.loads(fb.read_text()).get(name, {})
            for lab, v in human.items():
                if inf.get(lab, {}).get("status") == "pending_human":
                    inf[lab] = {**v, "judged_against": "blinded rating (A4/A4b)"}
        report["informative"] = inf

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
                     "cell_expert": "Mendeley_vmxhn95j8z", "cell_bus": "BUSClean", "cell_mg": "MedGemma"}.get(col)
        if gates.get(gate_name) != "analysed":
            continue
        if col in ("cell_bus", "cell_mg") and report.get("informative", {}).get(gate_name, {}).get("status") != "informative":
            continue  # model labellers count only when informative (A1)
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
