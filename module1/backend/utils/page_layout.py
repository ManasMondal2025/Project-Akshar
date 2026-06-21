"""
page_layout.py — Port of ScanTailor Advanced: PageLayout + Geometry
====================================================================
Direct port of:
  src/core/filters/page_split/PageLayout.cpp  (polygon clipping, outlines)
  src/core/imageproc/PolygonUtils.cpp          (Sutherland-Hodgman clipping)

Key exported symbols:
  LayoutType              — enum SINGLE_UNCUT | SINGLE_CUT | TWO_PAGES
  PageLayout              — geometric data model (outline + cutters)
  apply_perspective_warp  — warp a polygon region to a rectangle
  clip_polygon_by_halfplane — Sutherland-Hodgman
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

# ── Type aliases ──────────────────────────────────────────────────────────────
Point   = Tuple[float, float]   # (x, y) in image pixel coordinates
Polygon = List[Point]           # ordered list of (x, y)
Cutter  = Tuple[Point, Point]   # two points defining an infinite line


class LayoutType(str, Enum):
    SINGLE_UNCUT = "single_uncut"
    SINGLE_CUT   = "single_cut"
    TWO_PAGES    = "two_pages"


# ══════════════════════════════════════════════════════════════════════════════
# Low-level geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cross2d(v: Point, u: Point) -> float:
    """2-D cross product:  v × u = v.x·u.y − v.y·u.x"""
    return v[0] * u[1] - v[1] * u[0]


def _dot2d(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


# ── Length / area ─────────────────────────────────────────────────────────────

def polygon_bbox(polygon: Polygon) -> Tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max)"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(polygon: Polygon) -> float:
    """Shoelace formula — returns unsigned area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


# ── ScanTailor geometry helpers ───────────────────────────────────────────────

def extend_to_cover(cutter: Cutter, polygon: Polygon) -> Cutter:
    """
    Port of PageLayout::extendToCover().
    Extends cutter so that perpendiculars through its endpoints bound the polygon.
    Implemented by projecting all polygon vertices onto the cutter direction.
    """
    p1, p2 = cutter
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return cutter

    min_t = float('inf')
    max_t = float('-inf')
    for px, py in polygon:
        t = ((px - p1[0]) * dx + (py - p1[1]) * dy) / len_sq
        if t < min_t: min_t = t
        if t > max_t: max_t = t

    return (
        (p1[0] + min_t * dx, p1[1] + min_t * dy),
        (p1[0] + max_t * dx, p1[1] + max_t * dy),
    )


def ensure_same_direction(ref: Cutter, line: Cutter) -> Cutter:
    """
    Port of PageLayout::ensureSameDirection().
    Flips 'line' if its direction is opposite to 'ref'.
    """
    v_ref  = _sub(ref[1],  ref[0])
    v_line = _sub(line[1], line[0])
    if _dot2d(v_ref, v_line) < 0.0:
        return (line[1], line[0])
    return line


def clip_polygon_by_halfplane(
        polygon: Polygon, p1: Point, p2: Point, keep_right: bool = False
) -> Polygon:
    """
    Sutherland-Hodgman clipping of a (possibly non-convex) polygon
    by the half-plane of line p1→p2.

    keep_right=False (default): keeps points to the LEFT  (cross ≥ 0)
    keep_right=True:            keeps points to the RIGHT (cross ≤ 0)

    Port of PolygonUtils::clip() in Qt-based ScanTailor.
    """
    line_v = _sub(p2, p1)   # direction vector of clipping line

    def inside(pt: Point) -> bool:
        val = _cross2d(line_v, _sub(pt, p1))
        if keep_right:
            return val <= 0.0
        return val >= 0.0

    def intersect(q: Point, r: Point) -> Point:
        """Return intersection of line(p1,p2) with segment q→r."""
        seg_v = _sub(r, q)
        denom = seg_v[0] * line_v[1] - seg_v[1] * line_v[0]
        if abs(denom) < 1e-10:
            return q
        t = ((p1[0] - q[0]) * line_v[1] - (p1[1] - q[1]) * line_v[0]) / denom
        return (q[0] + t * seg_v[0], q[1] + t * seg_v[1])

    if not polygon:
        return []

    output: Polygon = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        prev = polygon[i - 1]
        if inside(curr):
            if not inside(prev):
                output.append(intersect(prev, curr))
            output.append(curr)
        elif inside(prev):
            output.append(intersect(prev, curr))

    return output


def order_polygon_as_quad(polygon: Polygon) -> Optional[Polygon]:
    """
    Extract 4 corner-representative points from a convex polygon as [TL, TR, BR, BL].
    Classification:
        TL → minimum  (x + y)
        TR → minimum  (y - x)
        BR → maximum  (x + y)
        BL → maximum  (y - x)
    """
    if len(polygon) < 4:
        return None
    pts = np.array(polygon, dtype=np.float64)
    tl = tuple(pts[np.argmin (pts[:, 0] + pts[:, 1])])
    tr = tuple(pts[np.argmin (pts[:, 1] - pts[:, 0])])
    br = tuple(pts[np.argmax (pts[:, 0] + pts[:, 1])])
    bl = tuple(pts[np.argmax (pts[:, 1] - pts[:, 0])])
    return [tl, tr, br, bl]


# ══════════════════════════════════════════════════════════════════════════════
# Core geometry-first operation
# ══════════════════════════════════════════════════════════════════════════════

def apply_perspective_warp(image: np.ndarray, polygon: Polygon) -> np.ndarray:
    """
    Warp 'image' so that the quadrilateral 'polygon' maps to an upright rectangle.

    This is the geometry-first split operation — instead of simple pixel-crop
    (image[:, x1:x2]) we use cv2.getPerspectiveTransform so the output is
    correctly rectified even for skewed / distorted scans.

    Output size is computed from the edge lengths of the polygon.
    """
    corners = order_polygon_as_quad(polygon)
    if corners is None:
        # Degenerate fallback: bounding-box crop
        x0, y0, x1, y1 = polygon_bbox(polygon)
        ih, iw = image.shape[:2]
        return image[max(0, int(y0)):min(ih, int(y1)),
                     max(0, int(x0)):min(iw, int(x1))]

    tl, tr, br, bl = corners

    # Output dimensions = max edge lengths for each pair of opposing sides
    w_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    w_bot = np.hypot(br[0] - bl[0], br[1] - bl[1])
    h_lft = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    h_rgt = np.hypot(br[0] - tr[0], br[1] - tr[1])

    dst_w = max(1, int(max(w_top, w_bot)))
    dst_h = max(1, int(max(h_lft, h_rgt)))

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]],
                   dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (dst_w, dst_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════════════
# PageLayout data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageLayout:
    """
    Port of ScanTailor's PageLayout (page_split/PageLayout.cpp).

    outline:     4-point polygon [TL, TR, BR, BL] in image pixel coordinates
    layout_type: LayoutType enum value
    cutter1:     primary cutter  — TWO_PAGES: the split line
                                 — SINGLE_CUT: left boundary cutter
    cutter2:     secondary cutter (SINGLE_CUT only — right boundary cutter)

    All geometry methods match the C++ implementation.
    """
    outline:     Polygon
    layout_type: LayoutType
    cutter1:     Optional[Cutter] = None
    cutter2:     Optional[Cutter] = None

    # ── Class-level constructors ──────────────────────────────────────────────

    @classmethod
    def from_image_size(cls, w: int, h: int) -> 'PageLayout':
        """Default full-image SINGLE_UNCUT layout."""
        outline = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
        return cls(outline=outline, layout_type=LayoutType.SINGLE_UNCUT)

    @classmethod
    def make_two_pages(cls, outline: Polygon, split_x: float) -> 'PageLayout':
        """TWO_PAGES with a vertical cutter at split_x pixel."""
        ys = [p[1] for p in outline]
        cutter: Cutter = ((split_x, min(ys) - 10.0), (split_x, max(ys) + 10.0))
        return cls(outline=outline, layout_type=LayoutType.TWO_PAGES, cutter1=cutter)

    @classmethod
    def make_two_pages_cutter(cls, outline: Polygon, cutter: Cutter) -> 'PageLayout':
        return cls(outline=outline, layout_type=LayoutType.TWO_PAGES, cutter1=cutter)

    @classmethod
    def make_single_cut(cls, outline: Polygon,
                        left_x: float, right_x: float) -> 'PageLayout':
        ys   = [p[1] for p in outline]
        ymin = min(ys) - 10.0
        ymax = max(ys) + 10.0
        c1: Cutter = ((left_x,  ymin), (left_x,  ymax))
        c2: Cutter = ((right_x, ymin), (right_x, ymax))
        return cls(outline=outline, layout_type=LayoutType.SINGLE_CUT, cutter1=c1, cutter2=c2)

    # ── ScanTailor geometry operations ────────────────────────────────────────

    def num_cutters(self) -> int:
        return {
            LayoutType.SINGLE_UNCUT: 0,
            LayoutType.SINGLE_CUT:   2,
            LayoutType.TWO_PAGES:    1,
        }[self.layout_type]

    def inscribed_cutter(self, idx: int = 0) -> Optional[Cutter]:
        """
        Port of PageLayout::inscribedCutterLine().
        Clips the stored infinite cutter to the polygon boundary.
        Returns the segment of the cutter that crosses the polygon.
        """
        cutter = self.cutter1 if idx == 0 else self.cutter2
        if cutter is None:
            return None

        p1, p2 = cutter
        line_v = _sub(p2, p1)
        len_sq = line_v[0] ** 2 + line_v[1] ** 2
        if len_sq < 1e-10:
            return None

        ts = []
        n = len(self.outline)
        for i in range(n):
            q1 = self.outline[i]
            q2 = self.outline[(i + 1) % n]
            seg_v = _sub(q2, q1)
            denom = line_v[0] * seg_v[1] - line_v[1] * seg_v[0]
            if abs(denom) < 1e-10:
                continue
            t = ((q1[0] - p1[0]) * seg_v[1] - (q1[1] - p1[1]) * seg_v[0]) / denom
            s = ((q1[0] - p1[0]) * line_v[1] - (q1[1] - p1[1]) * line_v[0]) / denom
            if -0.01 <= s <= 1.01:
                pt = (p1[0] + t * line_v[0], p1[1] + t * line_v[1])
                ts.append((t, pt))

        if len(ts) < 2:
            return cutter  # can't inscribe, return original

        ts.sort(key=lambda x: x[0])
        result: Cutter = (ts[0][1], ts[-1][1])
        return ensure_same_direction(cutter, result)

    def left_page_outline(self) -> Polygon:
        """
        Port of PageLayout::leftPageOutline() — TWO_PAGES only.
        Returns the polygon to the LEFT of cutter1.

        Implementation: clip the outline polygon by the half-plane
        to the left of the (top-to-bottom oriented) cutter line.
        """
        if self.layout_type != LayoutType.TWO_PAGES or not self.cutter1:
            return []

        # Extend the cutter to fully span the polygon
        p1, p2 = extend_to_cover(self.cutter1, self.outline)
        # Orient the cutter in the same direction as the left edge (TL→BL = downward)
        left_edge: Cutter = (self.outline[0], self.outline[3])
        p1, p2 = ensure_same_direction(left_edge, (p1, p2))

        # Sutherland-Hodgman: keep LEFT half-plane (content ← cutter)
        return clip_polygon_by_halfplane(self.outline, p1, p2, keep_right=False)

    def right_page_outline(self) -> Polygon:
        """
        Port of PageLayout::rightPageOutline() — TWO_PAGES only.
        Returns the polygon to the RIGHT of cutter1.
        """
        if self.layout_type != LayoutType.TWO_PAGES or not self.cutter1:
            return []

        p1, p2 = extend_to_cover(self.cutter1, self.outline)
        # Orient in same direction as right edge (TR→BR = downward)
        right_edge: Cutter = (self.outline[1], self.outline[2])
        p1, p2 = ensure_same_direction(right_edge, (p1, p2))

        # Sutherland-Hodgman: keep RIGHT half-plane (cutter → content)
        return clip_polygon_by_halfplane(self.outline, p1, p2, keep_right=True)

    def single_page_outline(self) -> Polygon:
        """
        Port of PageLayout::singlePageOutline().
        SINGLE_UNCUT  → full outline
        SINGLE_CUT    → outline clipped between cutter1 (left) and cutter2 (right)
        TWO_PAGES     → []  (not applicable for single page)
        """
        if self.layout_type == LayoutType.SINGLE_UNCUT:
            return list(self.outline)

        if self.layout_type == LayoutType.TWO_PAGES:
            return []

        # SINGLE_CUT — content is between cutter1 and cutter2
        if not self.cutter1 or not self.cutter2:
            return list(self.outline)

        ext1 = extend_to_cover(self.cutter1, self.outline)
        ext2 = extend_to_cover(self.cutter2, self.outline)
        ext2 = ensure_same_direction(ext1, ext2)

        # Keep RIGHT of left-cutter (to the right of cutter1)
        poly = clip_polygon_by_halfplane(self.outline, ext1[0], ext1[1], keep_right=True)
        # Keep LEFT of right-cutter (to the left of cutter2)
        poly = clip_polygon_by_halfplane(poly, ext2[0], ext2[1], keep_right=False)
        return poly

    def page_outline(self, side: str = 'single') -> Polygon:
        """Convenience accessor: 'left' | 'right' | 'single'."""
        if side == 'left':
            return self.left_page_outline()
        if side == 'right':
            return self.right_page_outline()
        return self.single_page_outline() or list(self.outline)

    # ── Validation (PageLayoutAdapter logic) ─────────────────────────────────

    def validate_and_fix(self) -> 'PageLayout':
        """
        Port of PageLayoutAdapter::adapt().
        Converts degenerate layouts (near-zero region) to SINGLE_UNCUT.
        """
        min_area = polygon_area(self.outline) * 0.05

        if self.layout_type == LayoutType.TWO_PAGES:
            if not self.cutter1:
                return PageLayout(self.outline, LayoutType.SINGLE_UNCUT)
            if polygon_area(self.left_page_outline()) < min_area:
                return PageLayout(self.outline, LayoutType.SINGLE_UNCUT)
            if polygon_area(self.right_page_outline()) < min_area:
                return PageLayout(self.outline, LayoutType.SINGLE_UNCUT)

        if self.layout_type == LayoutType.SINGLE_CUT:
            single = self.single_page_outline()
            if not single or polygon_area(single) < min_area:
                return PageLayout(self.outline, LayoutType.SINGLE_UNCUT)

        return self

    # ── Manual adjustment helpers ─────────────────────────────────────────────

    def with_cutter(self, idx: int, new_cutter: Cutter) -> 'PageLayout':
        """Return new layout with cutter idx replaced. Validates the result."""
        c1 = new_cutter if idx == 0 else self.cutter1
        c2 = new_cutter if idx == 1 else self.cutter2
        return PageLayout(self.outline, self.layout_type, c1, c2).validate_and_fix()

    def with_outline(self, new_outline: Polygon) -> 'PageLayout':
        """
        Port of PageLayoutAdapter: re-fit existing cutters to a new outline.
        Used when the user adjusts the page boundary corners.
        """
        if self.layout_type == LayoutType.SINGLE_UNCUT:
            return PageLayout(new_outline, LayoutType.SINGLE_UNCUT)

        # Scale cutters proportionally to new bounding box
        def _bbox(p): return polygon_bbox(p)
        ox0, oy0, ox1, oy1 = _bbox(self.outline)
        nx0, ny0, nx1, ny1 = _bbox(new_outline)
        ow, oh = max(ox1 - ox0, 1), max(oy1 - oy0, 1)
        nw, nh = nx1 - nx0, ny1 - ny0

        def _scale(pt: Point) -> Point:
            return (nx0 + (pt[0] - ox0) / ow * nw,
                    ny0 + (pt[1] - oy0) / oh * nh)

        def _scale_cutter(c: Optional[Cutter]) -> Optional[Cutter]:
            if c is None:
                return None
            return (_scale(c[0]), _scale(c[1]))

        return PageLayout(
            outline=new_outline,
            layout_type=self.layout_type,
            cutter1=_scale_cutter(self.cutter1),
            cutter2=_scale_cutter(self.cutter2),
        ).validate_and_fix()

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            'outline':     [list(p) for p in self.outline],
            'layout_type': self.layout_type.value,
            'cutter1':     [list(self.cutter1[0]), list(self.cutter1[1])]
                           if self.cutter1 else None,
            'cutter2':     [list(self.cutter2[0]), list(self.cutter2[1])]
                           if self.cutter2 else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PageLayout':
        lt = LayoutType(d['layout_type'])
        def _parse_cutter(raw) -> Optional[Cutter]:
            if raw is None:
                return None
            return (tuple(raw[0]), tuple(raw[1]))
        return cls(
            outline     = [tuple(p) for p in d['outline']],
            layout_type = lt,
            cutter1     = _parse_cutter(d.get('cutter1')),
            cutter2     = _parse_cutter(d.get('cutter2')),
        )

    # ── Inscribed cutter as ratio coordinates (for frontend) ─────────────────

    def inscribed_cutter_ratio(self, w: int, h: int, idx: int = 0
                               ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Returns the inscribed cutter as top/bottom ratio coords for the canvas UI.
        Format: {'top': {x: 0-1, y: 0-1}, 'bottom': {x: 0-1, y: 0-1}}
        """
        ic = self.inscribed_cutter(idx)
        if ic is None:
            return None
        top, bot = ic
        return {
            'top':    {'x': top[0] / w, 'y': top[1] / h},
            'bottom': {'x': bot[0] / w, 'y': bot[1] / h},
        }
