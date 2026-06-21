"""
ScanTailor Advanced — Page Split Detection (Faithful Python Port)
=================================================================
Direct port of the two-pass detection strategy from:
  - src/core/filters/page_split/PageLayoutEstimator.cpp
  - src/core/ContentSpanFinder.cpp
  - src/core/filters/page_split/VertLineFinder.cpp
  - src/core/filters/page_split/PageLayout.cpp

Pass 1 (primary):   VertLineFinder — Hough-based physical fold / gutter line detection
Pass 2 (fallback):  cutAtWhitespace — CC filter + ContentSpanFinder column-histogram analysis
"""

from __future__ import annotations

import base64
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .page_layout import (
    PageLayout, LayoutType,
    apply_perspective_warp, polygon_bbox,
)

# ── ContentSpanFinder constants (exact ScanTailor defaults) ──────────────────
_MIN_CONTENT_W   = 3      # minimum span width (columns) to be kept — scaled ×1.5 for 300 DPI
_MIN_WHITESPACE  = 12     # minimum gap width to break a content span — scaled ×1.5 for 300 DPI
_CC_MIN_DIM      = 8      # CC is noise if BOTH w<8 AND h<8 — scaled ×1.5 for 300 DPI
_CC_MAX_ASPECT   = 6.0    # CC is a binding crease if h/w > 6 (ratio — DPI independent)
_OFFCUT_MARGIN   = 3      # pixels from image edge (offcut strip margin) — scaled ×1.5
_OFFCUT_WIDTH    = 5      # width of the offcut probe strip — scaled ×1.5
_INSIG_DENOM     = 15     # edge span removable if width <= total / 15 (ratio — DPI independent)
_TWO_PAGE_THR    = 0.25   # if best gap ratio < this → treat as single span (ratio — DPI independent)
_ACCEPT_BAND     = 0.90   # widest gap among those with ratio ≥ 90 % of best (ratio — DPI independent)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


# ══════════════════════════════════════════════════════════════════════════════
# FAST SPINE DETECTION  (detect_spine_x / split_at)
# ══════════════════════════════════════════════════════════════════════════════
# Classical page splitting via projection profile and Hough analysis.
# This is the "Auto-Detect Spine" fast path used by the /split/detect-spine
# endpoint, complementing the full ScanTailor pipeline below.

def detect_spine_x(img: np.ndarray) -> int | None:
    """Locate the vertical spine line on a double-page spread.

    Runs two independent detectors — a projection-profile dip and a
    Hough-line vote — then returns their median when they agree, or
    falls back to whichever single signal is within tolerance.

    Returns the x-coordinate of the detected spine, or None if no
    confident detection is found.
    """
    gray = _to_gray(img)
    h, w = gray.shape
    cx = w // 2
    tolerance = int(w * 0.20)   # ±20 % of width around centre
    search_l = w // 4
    search_r = 3 * w // 4

    candidates: list[int] = []

    # ── Signal 1: projection-profile dip ─────────────────────────────────────
    # The gutter of a bound book casts a dark vertical shadow whose
    # column-mean is a local minimum in the central band.
    central = gray[:, search_l:search_r]
    col_means = central.mean(axis=0).astype(np.float32)
    if col_means.size >= 21:
        col_means = np.convolve(col_means, np.ones(21) / 21.0, mode="same")
    dip_local = int(np.argmin(col_means))
    dip_x = search_l + dip_local
    if abs(dip_x - cx) <= tolerance:
        candidates.append(dip_x)

    # ── Signal 2: Hough-line vote ─────────────────────────────────────────────
    # After edge detection, collect all near-vertical line segments that
    # fall in the central search band, then take the column they cluster
    # around most tightly (mode of their midpoint x-coordinates).
    edges = cv2.Canny(gray, threshold1=30, threshold2=100, apertureSize=3)
    edge_strip = np.zeros_like(edges)
    edge_strip[:, search_l:search_r] = edges[:, search_l:search_r]

    lines = cv2.HoughLinesP(
        edge_strip,
        rho=1,
        theta=np.pi / 180,
        threshold=int(h * 0.35),    # at least 35 % of image height
        minLineLength=int(h * 0.30),
        maxLineGap=int(h * 0.05),
    )
    if lines is not None:
        vertical_xs: list[int] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy == 0:
                continue
            # Keep only near-vertical segments (angle within 10° of 90°)
            if dx / dy < np.tan(np.radians(10)):
                mid_x = (x1 + x2) // 2
                if abs(mid_x - cx) <= tolerance:
                    vertical_xs.append(mid_x)

        if vertical_xs:
            xs = np.array(vertical_xs)
            bins = np.arange(search_l, search_r + 21, 20)
            hist, edges_b = np.histogram(xs, bins=bins)
            peak_bin = int(np.argmax(hist))
            hough_x = int((edges_b[peak_bin] + edges_b[peak_bin + 1]) / 2)
            if abs(hough_x - cx) <= tolerance:
                candidates.append(hough_x)

    if not candidates:
        return None

    # Two signals that agree → median is a reliable estimate.
    # One signal only → accept it (already passed the tolerance gate).
    return int(np.median(candidates))


def split_at(img: np.ndarray, x: int) -> tuple[np.ndarray, np.ndarray]:
    """Split image at column *x* into left and right halves."""
    return img[:, :x].copy(), img[:, x:].copy()


def _morphological_reconstruction(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Port of ScanTailor's seedFill(marker, mask, CONN8).
    Keeps every 8-connected component of `mask` that has at least one
    pixel overlapping with `marker`.
    """
    if not cv2.countNonZero(marker):
        return np.zeros_like(mask)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        cc = (labels == lbl).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(cc, marker)):
            result = cv2.bitwise_or(result, cc)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — VertLineFinder  (VertLineFinder.cpp::findLines)
# ══════════════════════════════════════════════════════════════════════════════

# A detected fold line segment in original image coordinates
_Segment = Tuple[Tuple[float, float], Tuple[float, float]]   # ((x1,y1),(x2,y2))


def _vert_line_finder(gray: np.ndarray) -> List[_Segment]:
    """
    Find near-vertical physical lines (book fold / gutter / binding shadow)
    using morphological preprocessing + probabilistic Hough transform.

    Port of VertLineFinder::findLines() [100 DPI version].
    Returns full line segments in original image coordinates, sorted left-to-right
    by midpoint x.  Preserving both endpoints allows the frontend to draw the
    line at the correct *angle* rather than always as a perfect vertical.
    """
    h, w = gray.shape
    if w < 10 or h < 10:
        return []

    # Downscale to ~100 DPI equivalent (target long side ≈ 500 px)
    scale = min(1.0, 500.0 / max(h, w))
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

    # Horizontal erosion (1×11) highlights horizontal variations.
    # Vertical erosion  (11×1) highlights vertical variations.
    # Difference  →  suppresses horizontal content, enhances vertical raster lines.
    # (ScanTailor: erodeGray + GRopClippedSubtract)
    h_eroded = cv2.erode(small, np.ones((1, 11), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    v_eroded = cv2.erode(small, np.ones((11, 1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    diff = cv2.subtract(v_eroded, h_eroded)          # saturated: negatives → 0

    # Vertical morphological close (1×19) connects short vertical segments.
    raster = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((19, 1), np.uint8))

    _, binary = cv2.threshold(raster, 15, 255, cv2.THRESH_BINARY)

    # Probabilistic Hough — only accept lines spanning ≥ 25 % of image height.
    min_len  = max(5, int(sh * 0.25))
    max_gap  = max(3, int(sh * 0.08))
    thresh   = max(10, int(sh * 0.15))
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180,
                             threshold=thresh,
                             minLineLength=min_len,
                             maxLineGap=max_gap)
    if lines is None:
        return []

    # Keep near-vertical lines: |dx/dy| < tan(30°) ≈ 0.577 (allows 60–120 degrees)
    max_slope = np.tan(np.deg2rad(30))
    segs: List[_Segment] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        if dy > 0 and (dx / dy) < max_slope:
            # Ensure top point has the smaller y (upward on image)
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            # Scale back to original image coordinates
            segs.append((
                (x1 / scale, y1 / scale),
                (x2 / scale, y2 / scale),
            ))

    if not segs:
        return []

    # Sort segments by midpoint x
    segs.sort(key=lambda s: (s[0][0] + s[1][0]) / 2.0)

    # Merge segments whose midpoint x is within 5 % of image width
    merge_thresh = w * 0.05
    merged: List[_Segment] = []
    for seg in segs:
        mx = (seg[0][0] + seg[1][0]) / 2.0
        if not merged:
            merged.append(seg)
        elif mx - (merged[-1][0][0] + merged[-1][1][0]) / 2.0 > merge_thresh:
            merged.append(seg)
        else:
            # Keep the longer segment of the two (better angle estimate)
            prev = merged[-1]
            prev_len = abs(prev[1][1] - prev[0][1])
            cur_len  = abs(seg[1][1]  - seg[0][1])
            if cur_len > prev_len:
                merged[-1] = seg

    # Remove segments within 3 % of either edge
    margin = w * 0.03
    return [s for s in merged
            if margin < (s[0][0] + s[1][0]) / 2.0 < (w - margin)]


def _seg_midpoint_x(seg: _Segment) -> float:
    return (seg[0][0] + seg[1][0]) / 2.0


def _seg_to_full_cutter(seg: _Segment, h: float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Extrapolate a detected segment so it spans the full image height [0, h].
    This ensures the cutter crosses the entire page boundary polygon when the
    detected segment only covers the middle portion of the image.
    """
    (x1, y1), (x2, y2) = seg
    dy = y2 - y1
    if abs(dy) < 1e-6:
        # Horizontal line — return as-is (shouldn't pass the slope filter)
        return seg
    # Parametric line:  P(t) = (x1,y1) + t*(dx,dy)
    dx = x2 - x1
    # t at y=0 and y=h
    t0 = (0 - y1) / dy
    t1 = (h - y1) / dy
    top = (x1 + t0 * dx, 0.0)
    bot = (x1 + t1 * dx, float(h))
    return (top, bot)


def _build_page_layout_from_seg(
        seg: _Segment, layout_type: str, cx1: int, cx2: int, w: int, h: int,
) -> Dict[str, Any]:
    """
    Build a PageLayout dict whose cutter1 is derived from the actual detected
    segment (preserving its angle), extended to span the full image height.
    """
    outline = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    cutter = _seg_to_full_cutter(seg, float(h))

    lt = LayoutType(layout_type) if layout_type in [e.value for e in LayoutType] \
         else LayoutType.SINGLE_UNCUT

    if lt == LayoutType.TWO_PAGES:
        pl = PageLayout(outline, LayoutType.TWO_PAGES, cutter1=cutter)
    elif lt == LayoutType.SINGLE_CUT:
        pl = PageLayout.make_single_cut(outline, float(cx1), float(cx2))
    else:
        pl = PageLayout(outline, LayoutType.SINGLE_UNCUT)

    return pl.to_dict()


def _classify_fold_lines(segs: List[_Segment], w: int, h: int,
                          layout_hint: str) -> Optional[Dict[str, Any]]:
    """
    Given detected fold-line segments (each with two endpoints), build a
    layout dict that includes a page_layout with a correctly angled cutter.

    Mirrors autoDetectTwoPageLayout / autoDetectSinglePageLayout.
    """
    # Work with midpoint x values for layout decisions
    xs = [_seg_midpoint_x(s) for s in segs]
    # Keep a mapping midpoint-x → segment for geometry reconstruction
    seg_by_x = {_seg_midpoint_x(s): s for s in segs}

    cx = w / 2.0

    # Decide number of pages
    if layout_hint == 'two_pages':
        num_pages = 2
    elif layout_hint in ('single_page', 'page_plus_offcut'):
        num_pages = 1
    else:  # 'auto'
        central = [x for x in xs if abs(x - cx) < 0.5 * cx]
        num_pages = 2 if central else 1

    boundary = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]
    base = dict(image_width=w, image_height=h,
                content_x1=0, content_x2=w,
                content_x1_ratio=0.0, content_x2_ratio=1.0,
                left_offcut=False, right_offcut=False,
                boundary=boundary)

    if num_pages == 2:
        if not xs:
            return None
        # BadTwoPageSplitter: reject lines where dist from centre > 60 % of half-width
        good = [x for x in xs if abs(x - cx) <= 0.6 * cx]
        if not good:
            good = xs
        split_x = min(good, key=lambda x: abs(x - cx))
        best_seg = seg_by_x[split_x]
        pl_dict  = _build_page_layout_from_seg(best_seg, 'two_pages', 0, w, w, h)
        return {**base,
                'layout_type':   'two_pages',
                'split_x':       float(split_x),
                'split_x_ratio': split_x / w,
                'confidence':    0.85,
                'page_layout':   pl_dict}

    else:  # single-page
        if not xs:
            return None   # fall through to Pass 2

        if len(xs) == 1:
            lx  = xs[0]
            seg = segs[0]
            if lx < cx:
                pl_dict = _build_page_layout_from_seg(seg, 'page_plus_offcut',
                                                       int(lx), w, w, h)
                return {**base,
                        'layout_type':     'page_plus_offcut',
                        'split_x':         float(lx),
                        'split_x_ratio':   lx / w,
                        'content_x1':      int(lx), 'content_x1_ratio': lx / w,
                        'left_offcut':     True,
                        'confidence':      0.75,
                        'page_layout':     pl_dict}
            else:
                pl_dict = _build_page_layout_from_seg(seg, 'page_plus_offcut',
                                                       0, int(lx), w, h)
                return {**base,
                        'layout_type':     'page_plus_offcut',
                        'split_x':         float(lx),
                        'split_x_ratio':   lx / w,
                        'content_x2':      int(lx), 'content_x2_ratio': lx / w,
                        'right_offcut':    True,
                        'confidence':      0.75,
                        'page_layout':     pl_dict}

        # Multiple lines → two-cutter layout (SINGLE_PAGE_CUT)
        l1, l2 = xs[0], xs[-1]
        pl_dict = _build_page_layout_from_seg(segs[0], 'single_cut',
                                               int(l1), int(l2), w, h)
        return {**base,
                'layout_type':     'page_plus_offcut',
                'split_x':         float((l1 + l2) / 2),
                'split_x_ratio':   ((l1 + l2) / 2) / w,
                'content_x1':      int(l1), 'content_x1_ratio': l1 / w,
                'content_x2':      int(l2), 'content_x2_ratio': l2 / w,
                'left_offcut':     True, 'right_offcut': True,
                'confidence':      0.75,
                'page_layout':     pl_dict}


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — cutAtWhitespace  (PageLayoutEstimator.cpp)
# ══════════════════════════════════════════════════════════════════════════════

def _binarize(image: np.ndarray) -> np.ndarray:
    """Grayscale + adaptive-Otsu binarisation  (text = 255, background = 0)."""
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _remove_garbage_and_downscale(binary: np.ndarray) -> np.ndarray:
    """
    Port of PageLayoutEstimator::removeGarbageAnd2xDownscale().

    1. 2× downscale (ReduceThreshold equivalent)
    2. Keep only pixels connected to ≥4 px horizontal OR vertical bars
       (port of openBrick + seedFill)
    3. Detect shadow regions (very long horizontal / vertical dark bands)
       and subtract them.

    Returns a cleaned binary at half the input resolution.
    """
    h, w = binary.shape

    # ── Step 1: 2× downscale ────────────────────────────────────────────────
    reduced = cv2.resize(binary, (max(1, w // 2), max(1, h // 2)),
                          interpolation=cv2.INTER_LINEAR)
    _, reduced = cv2.threshold(reduced, 127, 255, cv2.THRESH_BINARY)
    rh, rw = reduced.shape

    # ── Step 2: Keep content connected to ≥4 px bars ────────────────────────
    h_seed = cv2.morphologyEx(reduced, cv2.MORPH_OPEN, np.ones((1, 4), np.uint8))
    v_seed = cv2.morphologyEx(reduced, cv2.MORPH_OPEN, np.ones((4, 1), np.uint8))
    seed   = cv2.bitwise_or(h_seed, v_seed)
    cleaned = _morphological_reconstruction(seed, reduced)

    # ── Step 3: Remove shadow regions ───────────────────────────────────────
    # ScanTailor uses 200×14 (horizontal) and 14×300 (vertical) at 150 DPI.
    # Scale proportionally to actual image size.
    hor_w = max(20, min(200, rw // 3))
    hor_h = max(3,  min(14,  rh // 30))
    ver_h = max(50, min(300, rh // 2))
    ver_w = max(3,  min(14,  rw // 30))

    hor_seed = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                                 np.ones((hor_h, hor_w), np.uint8))
    ver_seed = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                                 np.ones((ver_h, ver_w), np.uint8))
    shadow_seed = cv2.bitwise_or(hor_seed, ver_seed)

    if cv2.countNonZero(shadow_seed) > 0:
        dilated = cv2.dilate(cleaned, np.ones((3, 3), np.uint8))
        shadows = _morphological_reconstruction(shadow_seed, dilated)
        cleaned = cv2.subtract(cleaned, shadows)

    return cleaned


def _cc_filter(binary: np.ndarray) -> np.ndarray:
    """
    Port of the ccImg construction in cutAtWhitespaceDeskewed150():
    • Skip CCs where BOTH w<5 AND h<5   (noise specks)
    • Skip CCs where h/w > 6            (tall thin binding crease)
    • For kept CCs: FILL BOUNDING RECT (not pixel mask) — exact ScanTailor behaviour.
    """
    out = np.zeros_like(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    for lbl in range(1, num_labels):
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if cw < _CC_MIN_DIM and ch < _CC_MIN_DIM:
            continue
        if cw > 0 and (ch / cw) > _CC_MAX_ASPECT:
            continue
        # Fill bounding rectangle (ScanTailor: ccImg.fill(cc.rect(), BLACK))
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        out[y:y + ch, x:x + cw] = 255
    return out


def _find_content_spans(col_hist: np.ndarray) -> List[Tuple[int, int]]:
    """
    Exact port of ContentSpanFinder::findImpl().
    col_hist[i] = number of black pixels in column i (integer).
    Returns list of (begin, end) with exclusive end, like Python slices.
    """
    n = int(len(col_hist))
    spans: List[Tuple[int, int]] = []

    # Initialisation trick from ContentSpanFinder.cpp (lines 15–16)
    content_end   = -_MIN_WHITESPACE
    content_begin = content_end
    i = 0

    while True:
        # Find next content column
        while i < n and col_hist[i] == 0:
            i += 1

        if i - content_end >= _MIN_WHITESPACE:
            if content_end - content_begin >= _MIN_CONTENT_W:
                spans.append((content_begin, content_end))
            content_begin = i

        if i == n:
            break

        # Find next whitespace column
        while i < n and col_hist[i] != 0:
            i += 1
        content_end = i

    if content_end - content_begin >= _MIN_CONTENT_W:
        spans.append((content_begin, content_end))

    return spans


def _remove_insignificant_edge_spans(
        spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Port of PageLayoutEstimator::removeInsignificantEdgeSpans().
    Peels leading / trailing spans that are smaller than total_width / 15.
    """
    if len(spans) <= 1:
        return list(spans)
    spans = list(spans)
    total = sum(e - b for b, e in spans)
    may_remove = total // _INSIG_DENOM

    while len(spans) > 1:
        fw = spans[0][1]  - spans[0][0]
        lw = spans[-1][1] - spans[-1][0]
        if fw < lw:
            if fw > may_remove:
                break
            may_remove -= fw
            spans.pop(0)
        else:
            if lw > may_remove:
                break
            may_remove -= lw
            spans.pop()
    return spans


def _check_offcut(col_hist: np.ndarray, side: str) -> bool:
    """Port of checkForLeftOffcut / checkForRightOffcut (150 DPI strip probe)."""
    m, wd = _OFFCUT_MARGIN, _OFFCUT_WIDTH
    if side == 'left':
        strip = col_hist[m: m + wd]
    else:
        n = len(col_hist)
        strip = col_hist[max(0, n - m - wd): max(0, n - m)]
    return bool(np.any(strip > 0))


def _process_single_span(span: Tuple[int, int], width: int) -> float:
    """
    Port of processTwoPagesWithSingleSpan().
    Decides where to place the split when only one content span exists
    in a two-page layout.
    """
    b, e = span
    page_center = width / 2.0
    box_center  = (b + e) / 2.0
    box_half    = (e - b) / 2.0
    dist = abs(page_center - box_center) - box_half

    if dist > 23:   # ~1.5mm gap at 300 DPI (was 15 px at 200 DPI)
        return page_center

    left_ws  = b
    right_ws = width - e
    if left_ws > right_ws:
        return float(max(0, b - 23))
    else:
        return float(min(width, e + 23))


def _process_two_pages(
        spans: List[Tuple[int, int]], width: int, height: int
) -> Tuple[float, int, int]:
    """
    Port of PageLayoutEstimator::processContentSpansTwoPages().
    Returns (split_x, content_x1, content_x2) in the working image coords.
    """
    if not spans:
        return width / 2.0, 0, width

    if len(spans) == 1:
        x = _process_single_span(spans[0], width)
        return x, 0, width

    content_begin = spans[0][0]
    content_end   = spans[-1][1]

    # Build gap balance ratios (ScanTailor lines 672–690)
    gaps: List[Tuple[int, int]] = []
    for i in range(len(spans) - 1):
        first  = spans[i][1]       - content_begin   # content left of gap
        second = content_end - spans[i + 1][0]       # content right of gap
        gaps.append((first, second))

    # Find best-balanced gap
    best_ratio = 0.0
    best_gap   = 0
    for i, (f, s) in enumerate(gaps):
        mx = max(f, s, 1)
        ratio = min(f, s) / mx
        if ratio > best_ratio:
            best_ratio = ratio
            best_gap   = i

    if best_ratio < _TWO_PAGE_THR:
        # One page nearly empty — treat full content as single span
        x = _process_single_span((content_begin, content_end), width)
        return x, 0, width

    # Find widest gap among those with ratio ≥ 90 % of best,
    # searching outward from bestGap and stopping on first failure.
    acceptable = best_ratio * _ACCEPT_BAND
    widest_gap = best_gap
    max_width  = spans[best_gap + 1][0] - spans[best_gap][1]   # gap width

    for i in range(best_gap - 1, -1, -1):               # search left
        f, s = gaps[i]
        if min(f, s) / max(max(f, s), 1) < acceptable:
            break
        gw = spans[i + 1][0] - spans[i][1]
        if gw > max_width:
            max_width = gw
            widest_gap = i

    for i in range(best_gap + 1, len(gaps)):             # search right
        f, s = gaps[i]
        if min(f, s) / max(max(f, s), 1) < acceptable:
            break
        gw = spans[i + 1][0] - spans[i][1]
        if gw > max_width:
            max_width = gw
            widest_gap = i

    gap_begin = spans[widest_gap][1]
    gap_end   = spans[widest_gap + 1][0]
    split_x   = (gap_begin + gap_end) / 2.0
    return split_x, content_begin, content_end


def _process_single_page(
        spans: List[Tuple[int, int]], width: int, height: int,
        left_offcut: bool, right_offcut: bool,
        layout_hint: str
) -> Tuple[str, float, int, int]:
    """
    Port of PageLayoutEstimator::processContentSpansSinglePage().
    Returns (layout_type, split_x, content_x1, content_x2).
    """
    # ── Left offcut branch ──────────────────────────────────────────────────
    if left_offcut and not right_offcut and layout_hint == 'auto':
        x = None
        if not spans:
            x = 0.0
        elif spans[0][0] > 0:
            x = 0.5 * spans[0][0]                        # midpoint of left gap
        else:
            sw = spans[0][1] - spans[0][0]
            if sw <= width // 2:
                if len(spans) > 1:
                    x = (spans[0][1] + spans[1][0]) / 2.0
                else:
                    x = float(min(spans[0][1] + 20, width))
        if x is not None:
            return 'page_plus_offcut', x, int(x), width

    # ── Right offcut branch ─────────────────────────────────────────────────
    if right_offcut and not left_offcut and layout_hint == 'auto':
        x = None
        if not spans:
            x = float(width)
        elif spans[-1][1] < width:
            x = (spans[-1][1] + width) / 2.0
        else:
            sw = spans[-1][1] - spans[-1][0]
            if sw <= width // 2:
                if len(spans) > 1:
                    x = (spans[-2][1] + spans[-1][0]) / 2.0
                else:
                    x = float(max(spans[-1][0] - 20, 0))
        if x is not None:
            return 'page_plus_offcut', x, 0, int(x)

    if layout_hint == 'page_plus_offcut':
        return 'page_plus_offcut', 0.0, 0, width

    return 'single_page', 0.0, 0, width


def _content_boundary(cc_img: np.ndarray, orig_w: int, orig_h: int,
                       scale_x: float, scale_y: float) -> List[List[float]]:
    """
    Compute the page content boundary as 4 corner-ratio pairs [TL,TR,BR,BL].
    Based on the bounding box of significant content CCs.
    """
    ys, xs = np.where(cc_img > 0)
    if len(ys) < 10:
        m = 0.02
        return [[m, m], [1 - m, m], [1 - m, 1 - m], [m, 1 - m]]

    m_px = 10  # small pixel margin
    x0 = max(0.0, (float(xs.min()) * scale_x - m_px) / orig_w)
    x1 = min(1.0, (float(xs.max()) * scale_x + m_px) / orig_w)
    y0 = max(0.0, (float(ys.min()) * scale_y - m_px) / orig_h)
    y1 = min(1.0, (float(ys.max()) * scale_y + m_px) / orig_h)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def detect_page_layout(image: np.ndarray,
                        layout_hint: str = 'auto') -> Dict[str, Any]:
    """
    Two-pass ScanTailor-faithful page layout detection.

    layout_hint: 'auto' | 'single_page' | 'two_pages' | 'page_plus_offcut'

    Returns a dict with:
      layout_type, split_x, split_x_ratio,
      content_x1, content_x2, content_x1_ratio, content_x2_ratio,
      confidence, image_width, image_height,
      left_offcut, right_offcut, boundary (4 corners as ratio pairs)
    """
    h, w = image.shape[:2]
    gray = _to_gray(image)

    # ── PASS 1: VertLineFinder ───────────────────────────────────────────────
    fold_lines = _vert_line_finder(gray)
    if fold_lines:
        result = _classify_fold_lines(fold_lines, w, h, layout_hint)
        if result is not None:
            return result

    # ── PASS 2: cutAtWhitespace fallback ─────────────────────────────────────

    # Binarize (≈300 DPI)
    binary = _binarize(image)

    # removeGarbageAnd2xDownscale → ≈150 DPI
    binary150 = _remove_garbage_and_downscale(binary)
    h2, w2 = binary150.shape

    # Column histogram for offcut probing (on raw 150 DPI)
    raw_hist = np.sum(binary150 // 255, axis=0).astype(np.int32)
    left_offcut  = _check_offcut(raw_hist, 'left')
    right_offcut = _check_offcut(raw_hist, 'right')

    # CC filter → column histogram
    cc_img   = _cc_filter(binary150)
    col_hist = np.sum(cc_img // 255, axis=0).astype(np.int32)

    # Scale factors from 150 DPI image back to original
    sx = w / max(w2, 1)
    sy = h / max(h2, 1)

    # Page content boundary (in original image ratio coords)
    boundary = _content_boundary(cc_img, w, h, sx, sy)

    # Content spans (exact ContentSpanFinder port)
    spans = _find_content_spans(col_hist)

    # Determine number of pages
    if layout_hint == 'two_pages':
        num_pages = 2
    elif layout_hint in ('single_page', 'page_plus_offcut'):
        num_pages = 1
    else:  # 'auto': landscape → probably 2 pages
        num_pages = 2 if (w > h * 1.15) else 1

    # ── Two-page path ────────────────────────────────────────────────────────
    if num_pages == 2:
        spans_r = _remove_insignificant_edge_spans(spans)
        split_x2, cx1_2, cx2_2 = _process_two_pages(spans_r, w2, h2)

        split_x = split_x2 * sx
        cx1     = int(cx1_2 * sx)
        cx2     = int(cx2_2 * sx)

        confidence = 0.75 if len(spans_r) >= 2 else 0.55
        return dict(
            layout_type      = 'two_pages',
            split_x          = float(split_x),
            split_x_ratio    = split_x / w,
            content_x1       = cx1,
            content_x2       = cx2,
            content_x1_ratio = cx1 / w,
            content_x2_ratio = cx2 / w,
            confidence       = confidence,
            image_width      = w,
            image_height     = h,
            left_offcut      = bool(left_offcut),
            right_offcut     = bool(right_offcut),
            boundary         = boundary,
        )

    # ── Single-page path ─────────────────────────────────────────────────────
    lt, split_x2, cx1_2, cx2_2 = _process_single_page(
        spans, w2, h2, left_offcut, right_offcut, layout_hint
    )
    split_x = split_x2 * sx
    cx1     = int(cx1_2 * sx)
    cx2     = int(cx2_2 * sx)

    return dict(
        layout_type      = lt,
        split_x          = float(split_x),
        split_x_ratio    = split_x / w,
        content_x1       = cx1,
        content_x2       = cx2,
        content_x1_ratio = cx1 / w,
        content_x2_ratio = cx2 / w,
        confidence       = 0.7,
        image_width      = w,
        image_height     = h,
        left_offcut      = bool(left_offcut),
        right_offcut     = bool(right_offcut),
        boundary         = boundary,
        # —— geometry-first: build PageLayout from detection result ——
        page_layout      = _make_page_layout(
            boundary, lt, split_x, cx1, cx2, w, h
        ).to_dict(),
    )



# ══════════════════════════════════════════════════════════════════════════════
# Geometry-first pipeline helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_page_layout(
        boundary: Optional[list],
        layout_type: str,
        split_x: float,
        cx1: int,
        cx2: int,
        w: int,
        h: int,
) -> PageLayout:
    """
    Build a PageLayout from detection results.
    'boundary' is the 4-corner list [[x,y], ...] in ratio coords returned by
    detect_page_layout(). If absent, uses the full image rectangle.
    """
    if boundary and len(boundary) >= 4:
        outline = [(p[0] * w, p[1] * h) for p in boundary]
    else:
        outline = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]

    lt = LayoutType(layout_type) if layout_type in [e.value for e in LayoutType] \
         else LayoutType.SINGLE_UNCUT

    if lt == LayoutType.TWO_PAGES:
        return PageLayout.make_two_pages(outline, split_x)
    elif lt == LayoutType.SINGLE_CUT:
        return PageLayout.make_single_cut(outline, float(cx1), float(cx2))
    else:
        return PageLayout(outline, LayoutType.SINGLE_UNCUT)


def _encode_img(img: np.ndarray) -> str:
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ''


# ── Geometry-first split (the NEW primary path) ───────────────────────────────

def split_page_geometry(
        image: np.ndarray,
        page_layout_dict: Dict[str, Any],
        selected_side: str = 'both',
) -> Dict[str, str]:
    """
    Geometry-first page split — port of ScanTailor's Task::process() pipeline.

    Pipeline:
        PageLayout → page_outline polygons → apply_perspective_warp → encode

    Instead of pixel-crop (image[:, x1:x2]) this function:
      1. Computes left / right / single page polygon via S-H clipping
      2. Applies a perspective warp so skewed pages come out upright
      3. Returns base64-encoded JPEG for each page region

    Args:
        image:            BGR numpy array
        page_layout_dict: dict from PageLayout.to_dict()
        selected_side:    'left' | 'right' | 'both' | 'single'

    Returns:
        dict with keys 'left', 'right', 'page' (present only when requested/applicable)
    """
    pl = PageLayout.from_dict(page_layout_dict)
    results: Dict[str, str] = {}

    if pl.layout_type == LayoutType.TWO_PAGES:
        if selected_side in ('left', 'both'):
            poly = pl.left_page_outline()
            if len(poly) >= 4:
                warped = apply_perspective_warp(image, poly)
                if warped.size > 0:
                    results['left'] = _encode_img(warped)

        if selected_side in ('right', 'both'):
            poly = pl.right_page_outline()
            if len(poly) >= 4:
                warped = apply_perspective_warp(image, poly)
                if warped.size > 0:
                    results['right'] = _encode_img(warped)

    else:  # SINGLE_UNCUT or SINGLE_CUT
        poly = pl.single_page_outline()
        if not poly:
            poly = list(pl.outline)
        if len(poly) >= 4:
            warped = apply_perspective_warp(image, poly)
            if warped.size > 0:
                results['page'] = _encode_img(warped)
        else:
            # Fallback: full image
            results['page'] = _encode_img(image)

    return results


# ── Pixel-crop fallback (kept for backward compatibility) ─────────────────────

def apply_split(
        image: np.ndarray,
        split_x: int,
        layout_type: str   = 'two_pages',
        content_x1: int    = 0,
        content_x2: int    = -1,
        selected_side: str = 'both',
) -> Dict[str, Any]:
    """
    LEGACY: axis-aligned pixel crop split.
    Prefer split_page_geometry() for correct perspective handling.
    """
    h, w = image.shape[:2]
    if content_x2 < 0:
        content_x2 = w

    split_x    = int(np.clip(split_x,    0, w))
    content_x1 = int(np.clip(content_x1, 0, w))
    content_x2 = int(np.clip(content_x2, 0, w))

    if layout_type == 'two_pages':
        results: Dict[str, str] = {}
        if selected_side in ('left', 'both') and split_x > content_x1:
            results['left'] = _encode_img(image[:, content_x1:split_x])
        if selected_side in ('right', 'both') and content_x2 > split_x:
            results['right'] = _encode_img(image[:, split_x:content_x2])
        return results
    else:
        cx1 = max(content_x1, 0)
        cx2 = content_x2 if content_x2 < w else w
        if cx2 <= cx1:
            cx2 = w
        return {'page': _encode_img(image[:, cx1:cx2])}

