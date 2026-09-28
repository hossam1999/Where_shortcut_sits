"""A0 — download sources, record licences, match cohorts, write the coverage report.

  python scripts/round6/sources.py
  python scripts/round6/sources.py --smoke
Licences are written to results/round6/sources.json before a source is used for matching.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from wtss import paths  # noqa: E402

TODAY = date.today().isoformat()
DERM_BYTES = 4139370781
ZENODO = {
    "derm": ("18324962", "DermArtifactDB.zip", "https://zenodo.org/records/18324962", "cc-by-4.0"),
    "ima": ("14201693", "segs.zip", "https://zenodo.org/records/14201693", "cc-by-nc-nd-4.0"),
}


def _wget(url: str, dest: Path):
    import subprocess
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        C.log(download="skip", file=dest.name, bytes=dest.stat().st_size)
        return
    C.log(download=url, dest=str(dest))
    subprocess.check_call(["curl", "-fL", "--retry", "3", "-o", str(dest), url])


def _zenodo_files(record: str) -> list[dict]:
    import urllib.request
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as r:
        d = json.loads(r.read().decode())
    lic = (d.get("metadata") or {}).get("license") or {}
    return d.get("files") or [], lic, (d.get("metadata") or {}).get("version")


def ensure_zenodo(kind: str):
    rec, _name, url, _lic = ZENODO[kind]
    files, lic, version = _zenodo_files(rec)
    C.EXT.mkdir(parents=True, exist_ok=True)
    got = []
    for f in files:
        dest = C.EXT / f["key"]
        if kind == "derm" and not f["key"].endswith(".zip"):
            continue
        if not dest.exists() or dest.stat().st_size != f["size"]:
            if dest.exists() and dest.stat().st_size < f["size"]:
                # a download may still be running; do not truncate it
                C.log(download="in_progress", file=f["key"], have=dest.stat().st_size, need=f["size"])
                return {"url": url, "version": version, "licence_id": lic.get("id"), "files": files, "ready": False}
            _wget(f["links"]["self"], dest)
        got.append(str(dest))
    return {"url": url, "version": version, "licence_id": lic.get("id"), "files": [f["key"] for f in files], "ready": True}


def read_busclean_licence() -> dict:
    root = C.EXT / "bus-cleaning"
    if not (root / "LICENSE").exists():
        import subprocess
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/hawaii-ai/bus-cleaning.git", str(root)])
    text = (root / "LICENSE").read_text()
    commit = os.popen(f"git -C {root} rev-parse HEAD").read().strip()
    permits = "Permission is hereby granted, free of charge, to any person obtaining a copy" in text
    return {"name": "BUSClean", "url": "https://github.com/hawaii-ai/bus-cleaning", "version": commit,
            "download_date": TODAY, "licence": "MIT", "licence_text": text.strip(),
            "research_use": bool(permits), "skip": not permits,
            "skip_reason": None if permits else "LICENSE does not grant research use",
            "detector": {
                "search": "grep -rni caliper over the bus-cleaning clone",
                "caliper_specific_function": None,
                "function_used": "detect_anno",
                "signature": "detect_anno(im: Image.Image, show_thresh: bool = False)",
                "also_defined": "detect_anno_BUSI(im: Image.Image, show_thresh: bool = False)",
                "note": "No function is named for calipers. README.md and SampleArtifacts.ipynb describe detect_anno, run after enhance_image, as the detector of lesion annotations, markers, and calipers. detect_anno_BUSI adds a Hough step for BUSI cross-style marks when detect_anno returns nothing; it is not used. No thresholds were changed. The registered validity guard decides on the full cohort.",
            }}


def medgemma_status() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    have = bool(token) or token_file.exists()
    return {"name": "MedGemma 1.5 4B-it", "url": "https://huggingface.co/google/medgemma-1.5-4b-it",
            "version": "google/medgemma-1.5-4b-it", "download_date": TODAY,
            "licence": "gated model terms (not accepted on this machine)" if not have else "model terms of use",
            "available": have,
            "skip_reason": None if have else "No Hugging Face token and no accepted-terms credential. "
            "The author must accept the model terms and run huggingface-cli login. This part is stopped; "
            "the rest of round 6 continues. No image is sent off the machine."}


def kabir_licence() -> dict:
    import urllib.request
    url = "https://data.mendeley.com/datasets/j5ywpd2p27/2"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    lic = "CC BY 4.0" if "CC BY 4.0" in html or "CC-BY" in html else "see dataset page"
    # quote the licence phrase actually present on the page
    idx = html.find("CC BY")
    quote = html[max(0, idx - 80):idx + 40].replace("\n", " ") if idx >= 0 else ""
    return {"name": "Kabir hair masks", "url": url, "version": "10.17632/j5ywpd2p27.2",
            "download_date": TODAY, "licence": lic, "licence_quote": quote,
            "local_path": str(paths.DATA / "mendeley_hair"),
            "note": "Already on the machine (Amendment 1). Licence read from the Mendeley dataset page."}


def mendeley_licence() -> dict:
    import urllib.request
    url = "https://data.mendeley.com/datasets/vmxhn95j8z/3"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    idx = html.find("CC BY 4.0")
    quote = ""
    if idx >= 0:
        quote = " ".join(html[max(0, idx - 220):idx + 12].split())
    return {"name": "Multicentre clear/contaminated capsule masks", "url": url, "version": "vmxhn95j8z v3",
            "download_date": TODAY, "licence": "CC BY 4.0" if idx >= 0 else "not found on page",
            "licence_quote": quote}


def isic2018_train_ids() -> set[str]:
    root = paths.DATA / "isic2018" / "ISIC2018_Task1-2_Training_Input"
    ids = set()
    if root.exists():
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                ids.add(C.isic_key(p.name))
    return ids


def figshare_probe_split():
    """Exactly scripts/data/prepare_capsule.expert_pairs and seed 20260927. Training frames are excluded."""
    sys.path.insert(0, str(paths.REPO_ROOT / "scripts" / "data"))
    import prepare_capsule as pc
    pairs = pc.expert_pairs()
    rng = np.random.default_rng(20260927)
    idx = rng.permutation(len(pairs))
    n_te = len(pairs) // 5
    te = [pairs[i] for i in idx[:n_te]]
    tr = [pairs[i] for i in idx[n_te:]]
    return pairs, te, tr


def _hash_u64(im: Image.Image) -> np.uint64:
    import imagehash
    h = imagehash.phash(im.convert("RGB").resize((256, 256)))
    bits = h.hash.astype(np.uint8).ravel()
    v = np.uint64(0)
    for b in bits:
        v = (v << np.uint64(1)) | np.uint64(int(b))
    return v


def popcount(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.uint64)
    c = np.zeros(x.shape, np.uint8)
    for s in range(64):
        c += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.uint8)
    return c


def extract_tables(zip_path: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    kept = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            low = name.lower()
            if info.is_dir():
                continue
            if low.endswith((".csv", ".tsv", ".json", ".txt", ".md", ".xlsx")) or "license" in low or "licence" in low:
                z.extract(info, dest)
                kept.append(name)
    return kept


def parse_derm(table_dir: Path, cohort_ids: set[str]) -> pd.DataFrame:
    frames = []
    for p in table_dir.rglob("*"):
        if p.suffix.lower() not in {".csv", ".tsv"}:
            continue
        try:
            d = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        except Exception:
            continue
        cols = {c.lower(): c for c in d.columns}
        idcol = next((cols[k] for k in cols if k in ("image_id", "isic_id", "img_id", "filename", "file", "image", "id")), None)
        hair = next((cols[k] for k in cols if k in ("hair", "hairs", "artifact_hair", "hair_present")), None)
        if idcol is None or hair is None:
            continue
        x = pd.DataFrame({"image_id": d[idcol].map(C.isic_key), "hair_present": pd.to_numeric(d[hair], errors="coerce")})
        x = x.dropna(subset=["hair_present"])
        x["hair_present"] = (x.hair_present > 0).astype(int)
        x = x[x.image_id.isin(cohort_ids)]
        frames.append(x)
        C.log(derm_table=str(p.name), rows=len(x))
    if not frames:
        return pd.DataFrame(columns=["image_id", "hair_present"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("image_id")


def ima_matched_ids(cohort_ids: set[str], exclude: set[str]) -> pd.DataFrame:
    meta = pd.read_csv(C.EXT / "ima_img_metadata.csv")
    meta["image_id"] = meta.isic_id.map(C.isic_key)
    m = meta[meta.image_id.isin(cohort_ids)].copy()
    m["excluded_isic2018_train"] = m.image_id.isin(exclude)
    return m[["image_id", "excluded_isic2018_train"]]


def kabir_index(cohort_ids: set[str]) -> pd.DataFrame:
    root = paths.DATA / "mendeley_hair"
    rows = []
    for p in (root.rglob("*.png")):
        if "hair_mask" not in str(p):
            continue
        k = C.isic_key(p.name)
        if k in cohort_ids:
            rows.append({"image_id": k, "mask_path": str(p)})
    return pd.DataFrame(rows)


def _seeai_pairs(root: Path):
    """Pair original frames with binary masks. Mendeley v3 uses Images/ and 'Binary  GT/'."""
    pairs = []
    image_dirs = []
    for imgs in root.rglob("*"):
        if imgs.is_dir() and imgs.name.lower() in {"imgs", "images", "img"}:
            image_dirs.append(imgs)
    for imgs in image_dirs:
        parent = imgs.parent
        mask_dir = None
        for cand in parent.iterdir() if parent.exists() else []:
            if not cand.is_dir():
                continue
            key = " ".join(cand.name.lower().split())
            if key in {"masks", "mask", "annotations", "binary gt"}:
                mask_dir = cand
                break
        if mask_dir is None:
            continue
        by_stem = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
        for im in sorted(imgs.iterdir()):
            if im.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            m = by_stem.get(im.stem)
            if m is not None:
                pairs.append((im, m))
    return pairs


def _extract_seeai(extracted: Path) -> bool:
    """Pull SEE-AI images and binary masks out of the Mendeley zip (it nests WCE dataset.zip)."""
    if (extracted / ".complete").exists():
        return True
    outer = C.EXT / "vmxhn95j8z-3.zip"
    inner_hits = list(C.EXT.rglob("WCE dataset.zip"))
    inner = inner_hits[0] if inner_hits else None
    if inner is None and outer.exists() and zipfile.is_zipfile(outer):
        with zipfile.ZipFile(outer) as z:
            z.extractall(C.EXT)
        outer.unlink(missing_ok=True)
        inner_hits = list(C.EXT.rglob("WCE dataset.zip"))
        inner = inner_hits[0] if inner_hits else None
    if inner is None or not zipfile.is_zipfile(inner):
        return False
    extracted.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(inner) as z:
        for name in z.namelist():
            if "SEE-AI" not in name or name.endswith("/"):
                continue
            if "/Images/" not in name and "Binary" not in name:
                continue
            kind = "masks" if "Binary" in name else "images"
            dest = extracted / "seeai" / kind / Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            n += 1
    if n == 0:
        return False
    (extracted / ".complete").write_text(f"seeai_files={n}\n")
    # the rest of the archive (Kvasir, CECleanliness, tri-colour copies) is not used again
    inner.unlink(missing_ok=True)
    parent = inner.parent
    if parent != C.EXT and parent.exists():
        import shutil
        shutil.rmtree(parent, ignore_errors=True)
    C.log(seeai_extracted=n)
    return True


def polarity_check(pairs, n=3) -> dict:
    """Registered convention: contaminated pixels are black (value < 128)."""
    rows = []
    for im, m in pairs[:n]:
        arr = np.asarray(Image.open(m).convert("L"))
        black = float((arr < 128).mean())
        rows.append({"image": im.name, "mask": m.name, "black_fraction": black, "shape": list(arr.shape)})
    ok = all(0.001 < r["black_fraction"] < 0.95 for r in rows) if rows else False
    return {"convention": "black<128 = contaminated", "n_checked": len(rows), "samples": rows,
            "accepted": bool(ok), "note": "Accepted when each of the three masks has a black fraction strictly between 0.001 and 0.95."}


def match_capsule(cohort: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, dict]:
    import imagehash
    extracted = C.EXT / "mendeley_capsule"
    if not (extracted / ".complete").exists():
        _extract_seeai(extracted)
    pairs, te, tr = figshare_probe_split()
    # training-set phashes (exclusion), cached
    cache_h = C.EXT / "figshare_train_phash.npy"
    if cache_h.exists():
        tr_h = np.load(cache_h)
    else:
        tr_paths = [p for _, p, _ in tr]
        if smoke:
            tr_paths = tr_paths[:30]
        hs = []
        for i, p in enumerate(tr_paths):
            hs.append(_hash_u64(Image.open(p)))
            if i and i % 500 == 0:
                C.log(figshare_train_hashed=i)
        tr_h = np.asarray(hs, np.uint64)
        if not smoke:
            np.save(cache_h, tr_h)
    if (extracted / ".complete").exists():
        see_root = extracted
    elif smoke:
        see_root = paths.DATA / "capsule" / "datasets"
    else:
        raise SystemExit("Mendeley vmxhn95j8z v3 SEE-AI masks were not extracted")
    see_pairs = _seeai_pairs(see_root)
    if smoke:
        see_pairs = see_pairs[:20]
    C.log(seeai_pairs=len(see_pairs), root=str(see_root))
    pol = polarity_check(see_pairs)
    if not pol["accepted"]:
        C.log(polarity="NOT accepted", detail=pol)
        return pd.DataFrame(columns=["image_id", "mask_path", "hamming"]), {"polarity": pol, "n_train_excluded_pool": int(len(tr)),
                                                                            "n_heldout_pool": int(len(te)), "matched": 0}
    # cohort hashes already stored
    coh = cohort[["image_id", "phash_hex"]].dropna()  # cohort keeps mask_src for the expert-label exclusion
    coh_bits = np.array([imagehash.hex_to_hash(h).hash.astype(np.uint8).ravel() for h in coh.phash_hex])
    coh_h = np.zeros(len(coh_bits), np.uint64)
    for s in range(64):
        coh_h |= coh_bits[:, s].astype(np.uint64) << np.uint64(63 - s)
    rows = []
    ids = coh.image_id.to_numpy()
    for im, m in see_pairs:
        h = _hash_u64(Image.open(im))
        d = popcount(np.bitwise_xor(coh_h, np.uint64(h)))
        j = int(np.argmin(d))
        if int(d[j]) <= 2:
            dtr = popcount(np.bitwise_xor(tr_h, np.uint64(h))) if len(tr_h) else np.array([99])
            excluded = bool(len(dtr) and int(dtr.min()) <= 2)
            rows.append({"image_id": ids[j], "mask_path": str(m), "image_path": str(im), "hamming": int(d[j]),
                         "excluded_probe_train": excluded})
    hit = pd.DataFrame(rows)
    if len(hit):
        hit = hit.sort_values("hamming").drop_duplicates("image_id")
    info = {"polarity": pol, "n_expert_pairs": len(pairs), "n_probe_train": len(tr), "n_probe_heldout": len(te),
            "n_seeai_files": len(see_pairs), "n_matched_before_exclusion": int((~hit.excluded_probe_train).sum()) + int(hit.excluded_probe_train.sum()) if len(hit) else 0,
            "n_excluded_probe_train": int(hit.excluded_probe_train.sum()) if len(hit) else 0}
    if len(hit):
        hit = hit[~hit.excluded_probe_train].drop(columns=["excluded_probe_train"])
        if "mask_src" in cohort.columns:  # frames whose own label is an expert mask: expert vs expert, excluded
            exp = set(cohort.loc[cohort.mask_src.astype(str) == "expert", "image_id"].astype(str))
            info["n_excluded_expert_labelled"] = int(hit.image_id.astype(str).isin(exp).sum())
            hit = hit[~hit.image_id.astype(str).isin(exp)]
    return hit, info


def _bus_one(job):
    name, image_id = job
    import sys as _sys
    _sys.path.insert(0, str(C.EXT / "bus-cleaning"))
    import modules.artifacts as A
    from modules.artifacts import enhance_image
    # prefer BUSClean's caliper detector; detect_anno may find text annotations, which the validity guard in
    # label_agreement.py then catches (positive on most caliper-free images -> invalid for calipers)
    detect = next((getattr(A, n) for n in ("detect_calipers", "detect_caliper", "find_calipers") if hasattr(A, n)), A.detect_anno)
    info = _bus_one.cache[name]
    rgb, roi, _ = info.get(str(image_id))
    boxes = detect(enhance_image(Image.fromarray(np.asarray(rgb))), False)
    present = len(boxes) > 0
    inside = False
    if present and roi is not None:
        H, W = roi.shape[:2]
        ih, iw = rgb.shape[:2]
        for box in boxes:
            if not box or not isinstance(box[0], (tuple, list)):
                continue
            cy = int(np.mean([p[1] for p in box]) * H / ih)
            cx = int(np.mean([p[0] for p in box]) * W / iw)
            if 0 <= cy < H and 0 <= cx < W and roi[cy, cx] > 0:
                inside = True
    cell = "artifact_free" if not present else ("trapA" if inside else "trapB")
    return {"image_id": str(image_id), "bus_present": int(present), "bus_inside": int(inside), "bus_cell": cell, "cohort": name}


def run_busclean(info, name: str, smoke: bool) -> pd.DataFrame:
    c = info["c"]
    if smoke:
        c = c.groupby("cell", group_keys=False).head(2)
    _bus_one.cache = {name: info["cache"]}
    jobs = [(name, i) for i in c.image_id]
    rows = C.R4.parallel_map(_bus_one, jobs)
    C.log(busclean=name, n=len(rows))
    return pd.DataFrame(rows)


def coverage_rows(cohort: str, source: str, matched: pd.DataFrame, c: pd.DataFrame) -> list[dict]:
    if matched is None or len(matched) == 0:
        return [{"cohort": cohort, "source": source, "cell": "all", "y": "all", "n": 0, "gate": "not_feasible"}]
    ids = set(matched.image_id.astype(str))
    sub = c[c.image_id.astype(str).isin(ids)]
    rows = []
    n = len(sub)
    rows.append({"cohort": cohort, "source": source, "cell": "all", "y": "all", "n": n, "gate": C.coverage_gate(n)})
    for cell in C.CELLS:
        for y in (0, 1):
            k = int(((sub.cell == cell) & (sub.y == y)).sum())
            rows.append({"cohort": cohort, "source": source, "cell": cell, "y": int(y), "n": k, "gate": C.coverage_gate(n)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = C.out_root(a.smoke)
    match_dir = out / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)
    sources = []

    bus = read_busclean_licence()
    sources.append({k: bus[k] for k in ("name", "url", "version", "download_date", "licence", "research_use", "skip", "skip_reason", "detector")})
    mg = medgemma_status()
    sources.append(mg)
    kab = kabir_licence()
    sources.append({k: kab[k] for k in ("name", "url", "version", "download_date", "licence", "licence_quote", "note")})
    men = mendeley_licence()
    sources.append(men)
    # write licences before matching
    (out / "sources.json").write_text(json.dumps({"sources": sources, "status": "licences_recorded_before_use"}, indent=2))
    C.log(stage="licences_written")

    derm_meta = ensure_zenodo("derm")
    ima_meta = ensure_zenodo("ima")
    sources.append({"name": "DermArtifactDB", "url": derm_meta["url"], "version": derm_meta["version"],
                    "download_date": TODAY, "licence": derm_meta["licence_id"],
                    "note": "Licence identifier read from the Zenodo record metadata before use. Labels only; the image zip is deleted after tables are extracted."})
    sources.append({"name": "IMA++", "url": ima_meta["url"], "version": ima_meta["version"],
                    "download_date": TODAY, "licence": ima_meta["licence_id"],
                    "note": "CC BY-NC-ND: measurement only. Masks and pixel-level derivatives stay under the data root and are never committed."})
    (out / "sources.json").write_text(json.dumps({"sources": sources, "status": "licences_recorded_before_use"}, indent=2))

    derm_zip = C.EXT / "DermArtifactDB.zip"
    derm_tab = C.EXT / "derm_tables"
    derm_ready = bool(derm_meta["ready"] and derm_zip.exists() and derm_zip.stat().st_size == DERM_BYTES)
    derm_tables_ready = derm_tab.exists() and any(derm_tab.rglob("*.csv"))
    if not a.smoke and not derm_ready and not derm_tables_ready:
        raise SystemExit("DermArtifactDB is still downloading. Re-run A0 when DermArtifactDB.zip is complete; nothing was scored.")

    frames = {name: C.cohort_frame(name) for name in C.COHORTS}
    isic = frames["isic"]["c"]
    isic_ids = set(isic.image_id.map(C.isic_key))
    ex2018 = isic2018_train_ids()

    # DermArtifactDB
    derm = pd.DataFrame(columns=["image_id", "hair_present"])
    if derm_meta["ready"] and derm_zip.exists() and derm_zip.stat().st_size == DERM_BYTES:
        if not any(derm_tab.rglob("*.csv")):
            kept = extract_tables(derm_zip, derm_tab)
            C.log(derm_tables=len(kept))
            if kept:
                derm_zip.unlink(missing_ok=True)
            else:
                C.log(derm="no annotation tables in the zip; zip kept for inspection")
        derm = parse_derm(derm_tab, isic_ids)
    elif (derm_tab.exists() and any(derm_tab.rglob("*.csv"))):
        derm = parse_derm(derm_tab, isic_ids)
    else:
        C.log(derm="zip not ready")
    if a.smoke and len(derm):
        derm = derm.head(30)
    derm.to_csv(match_dir / "isic_derm.csv", index=False)

    # IMA++
    ima = ima_matched_ids(isic_ids, ex2018) if (C.EXT / "ima_img_metadata.csv").exists() else pd.DataFrame(columns=["image_id", "excluded_isic2018_train"])
    if a.smoke and len(ima):
        ima = ima.head(30)
    ima.to_csv(match_dir / "isic_ima.csv", index=False)
    n_ima_ex = int(ima.excluded_isic2018_train.sum()) if len(ima) else 0
    ima_keep = ima[~ima.excluded_isic2018_train] if len(ima) else ima

    # Kabir
    kabir = kabir_index(isic_ids)
    if a.smoke and len(kabir):
        kabir = kabir.head(30)
    kabir.to_csv(match_dir / "isic_kabir.csv", index=False)

    # capsule
    cap_df, cap_info = match_capsule(frames["capsule"]["c"], a.smoke)
    cap_df.to_csv(match_dir / "capsule_expert.csv", index=False)
    (match_dir / "capsule_exclusion.json").write_text(json.dumps(cap_info, indent=2, default=str))

    # BUSClean on thyroid and ovary (MIT permits research use)
    bus_rows = []
    if not bus["skip"]:
        for name in ("thyroid", "ovary"):
            part = run_busclean(frames[name], name, a.smoke)
            part["cohort"] = name
            bus_rows.append(part)
            part.to_csv(match_dir / f"{name}_busclean.csv", index=False)
    bus_all = pd.concat(bus_rows, ignore_index=True) if bus_rows else pd.DataFrame()

    cov = []
    cov += coverage_rows("isic", "DermArtifactDB", derm, isic.assign(image_id=isic.image_id.map(C.isic_key)))
    cov += coverage_rows("isic", "IMA++", ima_keep, isic.assign(image_id=isic.image_id.map(C.isic_key)))
    cov += coverage_rows("isic", "Kabir", kabir, isic.assign(image_id=isic.image_id.map(C.isic_key)))
    cov += coverage_rows("capsule", "Mendeley_vmxhn95j8z", cap_df, frames["capsule"]["c"])
    if len(bus_all):
        for name in ("thyroid", "ovary"):
            cov += coverage_rows(name, "BUSClean", bus_all[bus_all.cohort == name], frames[name]["c"])
    else:
        cov += coverage_rows("thyroid", "BUSClean", pd.DataFrame(), frames["thyroid"]["c"])
        cov += coverage_rows("ovary", "BUSClean", pd.DataFrame(), frames["ovary"]["c"])
    cov += [{"cohort": "thyroid", "source": "MedGemma", "cell": "all", "y": "all", "n": 0,
             "gate": "not_feasible" if not mg["available"] else "pending"}]
    cov += [{"cohort": "ovary", "source": "MedGemma", "cell": "all", "y": "all", "n": 0,
             "gate": "not_feasible" if not mg["available"] else "pending"}]
    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(out / "coverage.csv", index=False)

    counts = {
        "isic2018_train_ids": len(ex2018),
        "ima_matched_before_exclusion": int(len(ima)),
        "ima_excluded_isic2018_train": n_ima_ex,
        "ima_analysed": int(len(ima_keep)),
        "derm_matched": int(len(derm)),
        "kabir_matched": int(len(kabir)),
        "capsule": {k: cap_info[k] for k in cap_info if k != "polarity"},
        "capsule_polarity_accepted": cap_info.get("polarity", {}).get("accepted"),
        "medgemma_available": mg["available"],
        "busclean_skipped": bus["skip"],
    }
    for s in sources:
        if s["name"] == "DermArtifactDB":
            s["n_matched"] = int(len(derm))
            readme = next((C.EXT / "derm_tables").rglob("README.md"), None)
            if readme:
                text = readme.read_text(errors="replace")
                i = text.find("Creative Commons")
                s["licence"] = "CC BY 4.0"
                s["licence_quote"] = " ".join(text[max(0, i - 80):i + 70].split()) if i >= 0 else ""
                s["note"] = "Licence read from README.md inside the Zenodo archive before the labels were used. Heatmaps were not extracted."
        elif s["name"] == "IMA++":
            s["n_matched"] = int(len(ima_keep))
            s["n_excluded_isic2018_train"] = n_ima_ex
        elif s["name"] == "Kabir hair masks":
            s["n_matched"] = int(len(kabir))
        elif s["name"].startswith("Multicentre"):
            s["n_matched"] = int(len(cap_df))
            s["n_excluded_probe_train"] = cap_info.get("n_excluded_probe_train")
        elif s["name"] == "BUSClean":
            s["n_matched"] = int(len(bus_all))
        elif s["name"].startswith("MedGemma"):
            s["n_matched"] = 0
    report = {"date": TODAY, "counts": counts, "sources": sources,
              "gates": cov_df[cov_df.cell == "all"][["cohort", "source", "n", "gate"]].to_dict(orient="records")}
    (out / "sources.json").write_text(json.dumps(report, indent=2))
    (out / "coverage.json").write_text(json.dumps(report["gates"], indent=2))
    C.log(stage="coverage", gates=report["gates"])
    print(cov_df[cov_df.cell == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
