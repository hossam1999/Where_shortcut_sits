"""Rule-based sonographer-marker segmentation for thyroid ultrasound (TN3K / TNCD).

Markers are (i) dotted measurement lines — small bright dots, evenly spaced along a straight line — and
(ii) small '+' / 'x' calipers at their ends. Tissue echoes are larger, smoother and not periodically collinear,
which is what the rules exploit. Frozen after tuning on one random sample (seed 3); validated by a visual audit
on fresh images (docs/THYROID_MARKER_AUDIT.md).
"""
from __future__ import annotations

import cv2
import numpy as np


def _dots(g: np.ndarray):
    """Small bright, *isolated* peaks (candidate marker dots): brighter than a 7x7 median and than every pixel of the
    surrounding ring (tissue echoes are part of larger bright structures and fail the ring test)."""
    gf = g.astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), 7).astype(np.float32)
    # ring mean (9x9 minus 5x5): robust to neighbouring dots of the same measurement line
    s9 = cv2.boxFilter(gf, -1, (9, 9), normalize=False)
    s5 = cv2.boxFilter(gf, -1, (5, 5), normalize=False)
    ring_mean = (s9 - s5) / (81 - 25)
    peak = ((gf - bg) > 32) & (gf > 105) & (gf - ring_mean > 45)
    n, lab, st, cen = cv2.connectedComponentsWithStats(peak.astype(np.uint8), 8)
    keep = [(cen[k], k) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] <= 9]
    return keep, lab


def _collinear_runs(pts: np.ndarray, min_n: int = 7, tol: float = 1.2, max_gap: float = 14.0):
    """Groups of >= min_n points lying on one line with near-regular spacing (dotted measurement lines)."""
    lines = []
    used = np.zeros(len(pts), bool)
    if len(pts) < min_n:
        return lines
    for i in range(len(pts)):
        if used[i]:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1:6]:
            if d[j] > max_gap or d[j] < 2:
                continue
            v = (pts[j] - pts[i]) / d[j]
            nrm = np.array([-v[1], v[0]])
            off = np.abs((pts - pts[i]) @ nrm)
            along = (pts - pts[i]) @ v
            cand = np.flatnonzero(off <= tol)
            if len(cand) < min_n:
                continue
            a = np.sort(along[cand])
            gaps = np.diff(a)
            # longest run with gaps in [2, max_gap] and roughly constant spacing
            good = (gaps >= 2) & (gaps <= max_gap)
            best, cur = 0, 0
            for gg in good:
                cur = cur + 1 if gg else 0
                best = max(best, cur)
            if best + 1 >= min_n and np.std(gaps[good]) <= 0.20 * np.median(gaps[good]):
                idx = cand[np.argsort(along[cand])]
                lines.append(idx)
                used[idx] = True
                break
    return lines


def _crosses(g: np.ndarray, shapes=("plus",)) -> np.ndarray:
    """Thin bright '+' shapes (caliper ends) by template matching on the top-hat image; optionally thick 'x' /
    asterisk calipers (GE Logiq style, ~11 px, 2-px arms) with their own thresholds."""
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    out = np.zeros(g.shape, np.uint8)
    for shape in shapes:
        for s in ((7, 9, 11) if shape == "plus" else (9, 11, 13)):
            t = np.zeros((s, s), np.float32)
            if shape == "plus":
                t[s // 2, :] = 1
                t[:, s // 2] = 1
                thr, tmin = 0.66, 60
            else:
                np.fill_diagonal(t, 1)
                np.fill_diagonal(np.fliplr(t), 1)
                t = cv2.dilate(t, np.ones((2, 2), np.uint8))
                thr, tmin = X_THR, X_TMIN
            tz = (t - t.mean()) / (t.std() + 1e-6)
            r = cv2.matchTemplate(th.astype(np.float32), tz, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.nonzero(r > thr)
            for y, x in zip(ys, xs):
                if th[y + s // 2, x + s // 2] > tmin:
                    out[y:y + s, x:x + s] |= (t > 0).astype(np.uint8)
    return out


X_THR, X_TMIN = 0.55, 40  # 'x' calipers: tuned on the BUS-BRA audit sample (seed 11), frozen before trap fitting


def marker_mask(gray: np.ndarray, shapes=("plus",)) -> np.ndarray:
    """Binary marker mask (dotted measurement lines + caliper crosses). shapes=("plus",) is the frozen thyroid
    detector (docs/THYROID_MARKER_AUDIT.md); breast ultrasound uses ("plus", "x") (docs/BREAST_MARKER_AUDIT.md)."""
    dots, lab = _dots(gray)
    m = np.zeros(gray.shape, np.uint8)
    if dots:
        pts = np.array([c for c, _ in dots])
        for run in _collinear_runs(pts):
            for k in run:
                m[lab == dots[k][1]] = 1
            a, b = pts[run[0]].astype(int), pts[run[-1]].astype(int)
            line = np.zeros_like(m)
            cv2.line(line, tuple(a), tuple(b), 1, 1)
            m |= line & (cv2.dilate((lab > 0).astype(np.uint8), np.ones((3, 3))) > 0)
    m |= _crosses(gray, shapes)
    m[: max(6, int(0.08 * m.shape[0])), :] = 0  # skin / fascia bands at the crop's top edge give dash-like echoes
    m[-6:, :] = 0
    return cv2.dilate(m, np.ones((3, 3), np.uint8))
