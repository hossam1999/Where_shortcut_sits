"""A4 — local blinded rating tool. Binds 127.0.0.1:8765 only.

  python scripts/round6/rating_app.py --pass 1                 # the 240-image audit package (audit/review_sheet.csv)
  python scripts/round6/rating_app.py --pass 1 --sheet a4b     # the A4b caliper sample (presence and location only)
  python scripts/round6/rating_app.py --pass 1 --check         # check every image file and exit (no port is bound)
Open with: ssh -p $VAST_TCP_PORT_22 -L 8765:127.0.0.1:8765 root@$PUBLIC_IPADDR
then http://localhost:8765
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHEETS = {"main": ROOT / "audit" / "review_sheet.csv", "a4b": ROOT / "results" / "round6" / "a4b_review_sheet.csv"}
OUT = ROOT / "results" / "round6" / "rating"
CONTOURS = ROOT / "audit_local" / "contours"  # raw image + expert ROI outline only (git-ignored, never committed)
COHORT = {"isic_hair": "isic", "thyroid_calipers": "thyroid", "ovary_calipers": "ovary", "capsule_debris": "capsule",
          "thyroid": "thyroid", "ovary": "ovary"}
WHAT = {
    "isic_hair": ("hair (dark thin strands)", "the skin lesion"),
    "thyroid_calipers": ("sonographer caliper or measurement marks (small + or x signs, dotted measurement lines)", "the nodule"),
    "ovary_calipers": ("sonographer caliper or measurement marks (small + or x signs, dotted measurement lines)", "the tumour"),
    "capsule_debris": ("bubbles, food debris, or turbid fluid", "the lesion area"),
    "thyroid": ("sonographer caliper or measurement marks (small + or x signs, dotted measurement lines)", "the nodule"),
    "ovary": ("sonographer caliper or measurement marks (small + or x signs, dotted measurement lines)", "the tumour"),
}


def make_contours():
    """Contour-only images for the main audit package: the green outline of the dataset's expert ROI, no automatic mask."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C
    ids = pd.read_csv(ROOT / "audit" / "sample_ids.csv")
    CONTOURS.mkdir(parents=True, exist_ok=True)
    caches = {}
    for r in ids.itertuples(index=False):
        f = CONTOURS / f"{r.audit_id}.png"
        if f.exists():
            continue
        name = COHORT[r.cohort]
        if name not in caches:
            caches[name] = C.cohort_frame(name)["cache"]
        rgb, roi, _ = caches[name].get(str(r.image_id))
        C.draw_contour(np.asarray(rgb), roi).save(f)
    return len(ids)
PASS_SEED = {1: 20260928, 2: 20261005}
SHEET_NAME = "main"


def load_order(pass_n: int, sheet: str = "main") -> pd.DataFrame:
    s = pd.read_csv(SHEETS[sheet])
    rng = np.random.default_rng(PASS_SEED[pass_n])
    order = rng.permutation(len(s))
    s = s.iloc[order].reset_index(drop=True)
    s["pos"] = np.arange(len(s))
    return s


def answers_path(pass_n: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / (f"pass{pass_n}.csv" if SHEET_NAME == "main" else f"a4b_pass{pass_n}.csv")


def read_answers(pass_n: int) -> dict:
    p = answers_path(pass_n)
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(p.open()):
        out[r["audit_id"]] = r
    return out


def write_answer(pass_n: int, row: dict, sheet: pd.DataFrame):
    fields = ["audit_id", "pass", "artifact_present", "artifact_location", "mask_correct", "roi_mask_correct", "notes"]
    p = answers_path(pass_n)
    cur = read_answers(pass_n)
    cur[row["audit_id"]] = row
    # stable order of the sheet, answers only
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for aid in sheet.audit_id:
            if aid in cur:
                w.writerow({k: cur[aid].get(k, "") for k in fields})


PAGE = """<!doctype html><meta charset=utf-8><title>Round 6 rating</title>
<style>
body {{ font: 16px/1.4 sans-serif; margin: 24px; max-width: 1400px; }}
img {{ max-width: 100%; background: #111; }}
.pair {{ display: flex; gap: 12px; align-items: flex-start; }}
.pair figure {{ flex: 1; margin: 0; min-width: 0; }}
.pair figcaption {{ font-size: 14px; color: #333; margin-bottom: 6px; }}
button {{ font-size: 16px; padding: 8px 14px; margin-right: 8px; }}
.step {{ color: #444; }}
</style>
<h1>Pass {pass_n} · {done}/{n}</h1>
<p class=step>Step {step} of {steps}. {aid}. The artifact for this image: <b>{what}</b>. Region: <b>{region}</b>
(green outline). The group this image belongs to is not shown.</p>
{images}
<form method=post action="/save">
<input type=hidden name=pos value="{pos}">
<input type=hidden name=step value="{step}">
{fields}
<p><label>Notes <input name=notes value="{notes}" size=60></label></p>
<button type=submit name=go value=next>Save and next</button>
</form>
"""


def form_fields(step, prev):
    if step == 1:
        pairs = [("artifact_present", "Artifact present?"), ("artifact_location", "Where is it relative to the lesion, nodule or tumour (outlined in green when shown)? (inside / outside / both / absent / unsure)")]
    else:
        pairs = [("mask_correct", "Is the artifact mask correct?"), ("roi_mask_correct", "Is the ROI mask correct?")]
    blocks = []
    for name, label in pairs:
        cur = html.escape(str(prev.get(name, "")))
        opts = "".join(f'<option value="{o}" {"selected" if cur==o else ""}>{o}</option>' for o in ("", "yes", "no", "unsure"))
        if name == "artifact_location":
            opts = "".join(f'<option value="{o}" {"selected" if cur==o else ""}>{o}</option>'
                           for o in ("", "inside", "outside", "both", "absent", "unsure"))
        blocks.append(f"<p><label>{label} <select name={name} required>{opts}</select></label></p>")
    return "".join(blocks)


class Handler(BaseHTTPRequestHandler):
    sheet = None
    pass_n = 1

    def log_message(self, fmt, *args):
        return

    def _img(self, pos, which):
        row = self.sheet.iloc[int(pos)]
        if which == "raw":
            path = ROOT / row.image_file
        elif which == "contour":  # a4b overlays are already contour-only
            path = (ROOT / row.overlay_file) if SHEET_NAME == "a4b" else CONTOURS / f"{row.audit_id}.png"
        else:
            path = ROOT / row.overlay_file
        rel = str(path.relative_to(ROOT))
        if not path.exists():
            self.send_error(404, str(rel))
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/img":
            self._img(q.get("pos", ["0"])[0], q.get("which", ["raw"])[0])
            return
        done = read_answers(self.pass_n)
        key = "artifact_present" if SHEET_NAME == "a4b" else "mask_correct"
        pending = [i for i, r in self.sheet.iterrows() if r.audit_id not in done or not done[r.audit_id].get(key)]
        # an image is finished only after step 2 (mask_correct saved)
        if not pending:
            body = f"<h1>Pass {self.pass_n} complete</h1><p>{len(done)} answers in {answers_path(self.pass_n)}</p>"
        else:
            pos = int(pending[0])
            row = self.sheet.iloc[pos]
            prev = done.get(row.audit_id, {})
            step = 1 if not prev.get("artifact_present") else 2
            what, region = WHAT.get(row.cohort, ("the artifact", "the region"))
            if step == 1:
                imgs = (f'<p>Image as the model saw it (left) and the same image with the region outlined in green '
                        f'(right).</p><img src="/img?pos={pos}&which=raw" alt="raw" style="max-width:49%"> '
                        f'<img src="/img?pos={pos}&which=contour" alt="outline" style="max-width:49%">')
            else:
                imgs = (f'<div class=pair>'
                        f'<figure><figcaption>Original image</figcaption>'
                        f'<img src="/img?pos={pos}&which=raw" alt="original"></figure>'
                        f'<figure><figcaption>Same image: green = region outline, red = automatic artifact mask</figcaption>'
                        f'<img src="/img?pos={pos}&which=overlay" alt="overlay"></figure>'
                        f'</div>')
            # step 1 shows the ROI outline for the location question: the A4b overlay has the outline only; for the
            # main package the raw image is shown first, as in audit/README.md
            body = PAGE.format(pass_n=self.pass_n, done=len(done), n=len(self.sheet), step=step, aid=html.escape(row.audit_id),
                               pos=pos, fields=form_fields(step, prev), notes=html.escape(str(prev.get("notes", ""))),
                               steps=1 if SHEET_NAME == "a4b" else 2, what=html.escape(what), region=html.escape(region),
                               images=imgs)
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        q = parse_qs(self.rfile.read(n).decode())
        pos = int(q["pos"][0])
        step = int(q["step"][0])
        row = self.sheet.iloc[pos]
        prev = read_answers(self.pass_n).get(row.audit_id, {"audit_id": row.audit_id, "pass": self.pass_n})
        prev["audit_id"] = row.audit_id
        prev["pass"] = str(self.pass_n)
        for k in ("artifact_present", "artifact_location", "mask_correct", "roi_mask_correct", "notes"):
            if k in q:
                prev[k] = q[k][0]
        if step == 1:
            prev.setdefault("mask_correct", "")
        write_answer(self.pass_n, prev, self.sheet)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_n", type=int, choices=(1, 2), default=1)
    ap.add_argument("--sheet", choices=("main", "a4b"), default="main")
    ap.add_argument("--smoke", action="store_true", help="build the sheet order and exit; do not bind a port")
    ap.add_argument("--check", action="store_true", help="check that every image file exists and exit; no port is bound")
    a = ap.parse_args()
    global SHEET_NAME
    SHEET_NAME = a.sheet
    if not SHEETS[a.sheet].exists():
        raise SystemExit(f"missing {SHEETS[a.sheet]}" + (" (run scripts/round6/a4b_sample.py)" if a.sheet == "a4b" else ""))
    sheet = load_order(a.pass_n, a.sheet)
    if a.sheet == "main" and not (a.smoke and not CONTOURS.exists()):
        n = make_contours()
        print(json.dumps({"contour_images": n, "dir": str(CONTOURS.relative_to(ROOT))}), flush=True)
    if a.smoke or a.check:
        files = list(sheet.image_file) + list(sheet.overlay_file)
        if a.sheet == "main":
            files += [str((CONTOURS / f"{i}.png").relative_to(ROOT)) for i in sheet.audit_id]
        missing = [f for f in files if not (ROOT / f).exists()]
        print(json.dumps({"sheet": a.sheet, "pass": a.pass_n, "n": len(sheet), "missing_files": len(missing),
                          "first_missing": missing[:3], "seed": PASS_SEED[a.pass_n],
                          "hint": "regenerate audit_local/ with python audit/make_audit_sample.py" if missing else ""}))
        return
    Handler.sheet = sheet
    Handler.pass_n = a.pass_n
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("rating tool on http://127.0.0.1:8765  pass", a.pass_n, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
