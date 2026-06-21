"""
Edge-based Corner Detection Utility
=====================================
Multi-strategy pipeline for detecting document corners in an image.

Strategies (tried in order):
  1. GrabCut foreground segmentation
  2. Saturation + Brightness thresholding
  3. Canny edge detection
  4. minAreaRect fallback

If all strategies fail, returns a default 8% inset rectangle.

Public API:
  - detect_corners(image)        → [[x,y], [x,y], [x,y], [x,y]]  (TL, TR, BR, BL)
  - warp_from_corners(image, corners) → warped np.ndarray
"""

import cv2
import numpy as np

_DOWNSCALE = 600
_MIN_AREA = 0.10
_FRAME_TOL = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_corners(image: np.ndarray) -> list[list[int]]:
    """
    Detect the four document corners in the given image.

    Returns corner coordinates as [[x,y], ...] in TL, TR, BR, BL order,
    mapped back to the original image space.
    """
    oh, ow = image.shape[:2]
    work, scale = _downscale(image)
    wh, ww = work.shape[:2]

    quad = _grabcut_quad(work, ww, wh)
    if quad is None:
        quad = _sat_boundary(work, ww, wh)
    if quad is None:
        quad = _canny_boundary(work, ww, wh)
    if quad is None:
        quad = _minrect_fallback(work, ww, wh)

    if quad is None:
        ix, iy = int(ow * 0.08), int(oh * 0.08)
        return [[ix, iy], [ow - ix, iy], [ow - ix, oh - iy], [ix, oh - iy]]

    pts = (_order(quad) * scale).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, ow)
    pts[:, 1] = np.clip(pts[:, 1], 0, oh)
    return [[int(p[0]), int(p[1])] for p in pts]


def warp_from_corners(image: np.ndarray, corners: list) -> np.ndarray:
    """
    Apply a perspective warp using 4 corner points [TL, TR, BR, BL].

    Returns the rectified image.
    """
    return _warp(image, np.array(corners, dtype=np.float32))


# ---------------------------------------------------------------------------
# Strategy 1 – GrabCut
# ---------------------------------------------------------------------------

def _grabcut_quad(work, fw, fh):
    mx = max(2, int(fw * 0.05))
    my = max(2, int(fh * 0.05))
    rect = (mx, my, fw - 2 * mx, fh - 2 * my)
    mask = np.zeros((fh, fw), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(work, mask, rect, bgd_model, fgd_model,
                    iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    fgd_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    k = max(5, min(15, fh // 50))
    if k % 2 == 0:
        k += 1
    ker = np.ones((k, k), np.uint8)
    fgd_mask = cv2.morphologyEx(fgd_mask, cv2.MORPH_CLOSE, ker, iterations=2)
    fgd_mask = cv2.morphologyEx(fgd_mask, cv2.MORPH_OPEN, ker, iterations=1)

    cnts, _ = cv2.findContours(fgd_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best, best_score = None, -1.0
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(cnt) / (fw * fh) < _MIN_AREA:
            continue
        quad = _hull_to_quad(cnt, fw * fh)
        if quad is None:
            continue
        s = _score(quad, fw, fh)
        if s > best_score:
            best_score, best = s, quad
    return best


# ---------------------------------------------------------------------------
# Strategy 2 – Saturation + Brightness
# ---------------------------------------------------------------------------

def _sat_boundary(work, fw, fh):
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    best, best_score = None, -1.0

    for s_thr, v_thr in [(40, 90), (50, 80), (35, 100), (60, 70)]:
        _, s_m = cv2.threshold(s_ch, s_thr, 255, cv2.THRESH_BINARY_INV)
        _, v_m = cv2.threshold(v_ch, v_thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(s_m, v_m)

        k = max(5, min(15, fh // 60))
        if k % 2 == 0:
            k += 1
        ker = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) / (fw * fh) < _MIN_AREA:
                continue
            quad = _hull_to_quad(cnt, fw * fh)
            if quad is None:
                continue
            s = _score(quad, fw, fh)
            if s > best_score:
                best_score, best = s, quad
    return best


# ---------------------------------------------------------------------------
# Strategy 3 – Canny Edge Detection
# ---------------------------------------------------------------------------

def _canny_boundary(work, fw, fh):
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    fa = fw * fh
    best, best_score = None, -1.0

    for bk, lo, hi in [(5, "auto", "auto"), (7, "auto", "auto"),
                        (5, 25, 100), (5, 50, 150)]:
        blr = cv2.GaussianBlur(gray, (bk, bk), 0)
        if lo == "auto":
            m = float(np.median(blr))
            lo = int(max(0, 0.66 * m))
            hi = int(min(255, 1.33 * m))
        edges = cv2.Canny(blr, lo, hi)

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=2)

        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(cnt) / fa < _MIN_AREA:
                continue
            quad = _hull_to_quad(cnt, fa)
            if quad is None:
                continue
            s = _score(quad, fw, fh)
            if s > best_score:
                best_score, best = s, quad
    return best


# ---------------------------------------------------------------------------
# Strategy 4 – minAreaRect Fallback
# ---------------------------------------------------------------------------

def _minrect_fallback(work, fw, fh):
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    _, s_m = cv2.threshold(hsv[:, :, 1], 55, 255, cv2.THRESH_BINARY_INV)
    _, v_m = cv2.threshold(hsv[:, :, 2], 85, 255, cv2.THRESH_BINARY)
    comb = cv2.bitwise_and(s_m, v_m)

    k = max(7, min(21, fh // 40))
    if k % 2 == 0:
        k += 1
    ker = np.ones((k, k), np.uint8)
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, ker, iterations=2)

    cnts, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    lg = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(lg) / (fw * fh) < _MIN_AREA:
        return None

    box = cv2.boxPoints(cv2.minAreaRect(lg)).astype(np.float32)
    return box if _score(box, fw, fh) > 0.05 else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(quad, fw, fh):
    fa = fw * fh
    qa = float(cv2.contourArea(quad.astype(np.int32)))
    rat = qa / fa
    if rat < _MIN_AREA:
        return 0.0

    _, (rw, rh), _ = cv2.minAreaRect(quad.astype(np.int32))
    rect_s = min(1.0, qa / (rw * rh)) if rw * rh > 0 else 0.0

    on_bd = sum(
        1 for pt in quad
        if float(pt[0]) < _FRAME_TOL or float(pt[0]) > fw - _FRAME_TOL
        or float(pt[1]) < _FRAME_TOL or float(pt[1]) > fh - _FRAME_TOL
    )
    bd_pen = [1.0, 1.0, 0.85, 0.35, 0.08][min(on_bd, 4)]
    return rat * rect_s * bd_pen


# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------

def _downscale(image):
    h, w = image.shape[:2]
    lng = max(h, w)
    if lng <= _DOWNSCALE:
        return image.copy(), 1.0
    sc = lng / _DOWNSCALE
    work = cv2.resize(image, (int(w / sc), int(h / sc)),
                      interpolation=cv2.INTER_AREA)
    return work, sc


def _hull_to_quad(cnt, frame_area):
    hull = cv2.convexHull(cnt)
    if len(hull) < 4:
        return None
    peri = cv2.arcLength(hull, True)
    if peri == 0:
        return None
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                0.08, 0.10, 0.13, 0.16, 0.20]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) / frame_area >= _MIN_AREA:
                return approx.reshape(4, 2).astype(np.float32)
    return _quadrant_pts(hull)


def _quadrant_pts(hull):
    pts = hull.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    best, bdist = [None] * 4, [0.0] * 4
    for p in pts:
        d = float(np.hypot(p[0] - cx, p[1] - cy))
        q = (0 if p[0] < cx and p[1] < cy else
             1 if p[0] >= cx and p[1] < cy else
             2 if p[0] >= cx else 3)
        if d > bdist[q]:
            bdist[q], best[q] = d, p.copy()
    return None if any(b is None for b in best) else np.array(best, np.float32)


def _order(pts):
    pts = np.array(pts, dtype=np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).flatten()
    r = np.zeros((4, 2), dtype=np.float32)
    r[0] = pts[np.argmin(s)]
    r[2] = pts[np.argmax(s)]
    r[1] = pts[np.argmin(d)]
    r[3] = pts[np.argmax(d)]
    return r


def _warp(image, ordered):
    tl, tr, br, bl = ordered
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w <= 0 or h <= 0:
        return image
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
