"""
Module 1 – ScanTailor-Advanced Style Cylindrical-Surface Mesh Dewarping
========================================================================
A faithful Python re-implementation (using NumPy / SciPy / OpenCV) of the
dewarping algorithm from ScanTailor Advanced.

Algorithm overview
------------------
1. Detect text-line polylines in the document image.
2. Select the best TOP and BOTTOM bounding curves (directrices) using a
   RANSAC-like scoring procedure.
3. Build a *CylindricalSurfaceDewarper* from the two directrices:
   • 4-point 2-D homography maps the quadrilateral formed by directrix
     endpoints onto a unit square (the "plane" coordinate system).
   • Arc-length mapping along the coupled directrices compensates for
     foreshortening at curved regions.
   • For every output-column *x_crv* a *generatrix* (vertical line in image
     space connecting the two directrices) is computed, together with a 1-D
     homographic mapping that distributes pixels evenly along it.
4. The final image is produced by ``cv2.remap`` on the computed mesh grid.

Reference: ``scantailor-advanced-master/src/dewarping/`` (C++) – used only
as an algorithmic blueprint; no C++ code is linked or imported.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import interpolate as sp_interp
from typing import List, Tuple, Optional


# ===================================================================
#  Low-level math helpers (ported from ScanTailor C++ helpers)
# ===================================================================

def _four_point_homography_2d(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """
    Compute the 3×3 homography H such that  dst ≈ H @ src  (homogeneous coords).

    Parameters
    ----------
    src : (4, 2) array – four source points
    dst : (4, 2) array – four destination points

    Returns
    -------
    H : (3, 3) array
    """
    # Use OpenCV's findHomography for robustness
    H, _ = cv2.findHomography(src.astype(np.float64),
                               dst.astype(np.float64))
    if H is None:
        # Fallback: identity
        return np.eye(3, dtype=np.float64)
    return H


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 homography to an array of 2-D points.

    Parameters
    ----------
    H   : (3, 3)
    pts : (N, 2) or (2,)

    Returns
    -------
    (N, 2) or (2,) – transformed points
    """
    squeeze = pts.ndim == 1
    pts = np.atleast_2d(pts).astype(np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts, ones])            # (N, 3)
    out = (H @ hom.T).T                     # (N, 3)
    out = out[:, :2] / out[:, 2:3]          # perspective divide
    if squeeze:
        return out[0]
    return out


def _three_point_1d_homography(pairs: np.ndarray) -> np.ndarray:
    """
    Compute a 1-D homographic transform from three (from, to) pairs.

    The transform is:  to = (a * from + b) / (c * from + 1)

    Parameters
    ----------
    pairs : (3, 2) – each row is (from_val, to_val)

    Returns
    -------
    coeffs : (3,) array [a, b, c]
    """
    A = np.zeros((3, 3), dtype=np.float64)
    B = np.zeros(3, dtype=np.float64)
    for i in range(3):
        f, t = pairs[i]
        A[i] = [-f, -1.0, f * t]
        B[i] = -t
    try:
        x = np.linalg.solve(A, B)  # [a, b, c]
    except np.linalg.LinAlgError:
        x = np.array([1.0, 0.0, 0.0])
    return x


def _apply_1d_homography(coeffs: np.ndarray, val: float) -> float:
    """Apply 1-D homographic transform: (a*x + b) / (c*x + 1)."""
    a, b, c = coeffs
    denom = c * val + 1.0
    if abs(denom) < 1e-15:
        return val
    return (a * val + b) / denom


def _polyline_length(pts: np.ndarray) -> float:
    """Total chord length of a polyline (N, 2)."""
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def _intersect_line_polyline(
    line_pt: np.ndarray,
    line_dir: np.ndarray,
    polyline: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Find the intersection of an infinite line with a polyline.

    Parameters
    ----------
    line_pt  : (2,) – a point on the line
    line_dir : (2,) – direction vector of the line
    polyline : (N, 2) – polyline vertices

    Returns
    -------
    intersection : (2,) – point of intersection
    t            : float – parameter along the line
    """
    best_pt = polyline[0].copy()
    best_dist = np.inf

    # Line as ax + by + c = 0
    normal = np.array([-line_dir[1], line_dir[0]], dtype=np.float64)
    c = -normal.dot(line_pt)

    N = len(polyline)
    for i in range(N - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        d1 = normal.dot(p1) + c
        d2 = normal.dot(p2) + c
        if d1 * d2 <= 0 and abs(d1 - d2) > 1e-12:
            t_seg = d1 / (d1 - d2)
            pt = p1 + t_seg * (p2 - p1)
            dist = np.abs(np.cross(line_dir, pt - line_pt))
            if dist < best_dist:
                best_dist = dist
                best_pt = pt

    t = 0.0
    if np.dot(line_dir, line_dir) > 1e-12:
        t = np.dot(best_pt - line_pt, line_dir) / np.dot(line_dir, line_dir)
    return best_pt, t


def _project_onto_line(line_p1: np.ndarray, line_p2: np.ndarray,
                        pt: np.ndarray) -> float:
    """
    Project *pt* onto the infinite line through *line_p1* → *line_p2*.
    Returns the scalar t such that  projection = p1 + t*(p2 - p1).
    """
    d = line_p2 - line_p1
    len_sq = np.dot(d, d)
    if len_sq < 1e-15:
        return 0.0
    return float(np.dot(pt - line_p1, d) / len_sq)


# ===================================================================
#  ArcLengthMapper
# ===================================================================

class _ArcLengthMapper:
    """
    Maps between *plane X* and *arc-length X* along coupled directrices.

    Conceptually, it walks both directrices simultaneously, measuring how
    much each generatrix "rises" above the flat plane (elevation), and
    accumulates arc-length as  dl = sqrt(dx² + delevation²).

    After construction, `arc_len_to_x(crv_x)` converts an equal-arc-length
    column position to the corresponding plane-X, and vice versa.
    """

    def __init__(self):
        self._samples_x: List[float] = []
        self._samples_arc: List[float] = []
        self._total_arc: float = 0.0
        # interpolators (set after normalise)
        self._arc_to_x = None
        self._x_to_arc = None

    def add_sample(self, pln_x: float, elevation: float):
        if self._samples_x:
            prev_x = self._samples_x[-1]
            dx = pln_x - prev_x
            de = elevation - self._samples_arc[-1]   # reuse field temporarily
            dl = np.sqrt(dx * dx + de * de)
            self._total_arc += dl
            self._samples_arc[-1] = self._total_arc - dl  # fix previous
        self._samples_x.append(pln_x)
        self._samples_arc.append(self._total_arc)

    def build(self, depth_perception: float,
              pln2img, img2pln,
              directrix1: np.ndarray, directrix2: np.ndarray):
        """
        Walk both directrices and build arc-length samples.
        """
        self._samples_x.clear()
        self._samples_arc.clear()
        self._total_arc = 0.0

        pts1, pts2, pln_xs = _coupled_polylines(
            directrix1, directrix2, pln2img, img2pln
        )

        prev_elevation = 0.0
        prev_pln_x = -1e18

        for i in range(len(pln_xs)):
            px = pln_xs[i]
            if px <= prev_pln_x:
                continue  # skip S-shaped regions

            p1 = pts1[i]
            p2 = pts2[i]
            img_gen_top = _apply_homography(pln2img, np.array([px, 0.0]))
            img_gen_bot = _apply_homography(pln2img, np.array([px, 1.0]))
            gen_dir = img_gen_bot - img_gen_top
            gen_len_sq = np.dot(gen_dir, gen_dir)
            if gen_len_sq < 1e-12:
                continue

            y1 = np.dot(p1 - img_gen_top, gen_dir) / gen_len_sq
            y2 = np.dot(p2 - img_gen_top, gen_dir) / gen_len_sq

            elevation = depth_perception * (1.0 - (y2 - y1))
            elevation = max(-0.5, min(0.5, elevation))

            # Arc-length increment
            if self._samples_x:
                dx = px - self._samples_x[-1]
                de = elevation - prev_elevation
                dl = np.sqrt(dx * dx + de * de)
                self._total_arc += dl
            self._samples_x.append(px)
            self._samples_arc.append(self._total_arc)

            prev_elevation = elevation
            prev_pln_x = px

        if len(self._samples_x) < 2:
            # Degenerate – linear identity
            self._samples_x = [0.0, 1.0]
            self._samples_arc = [0.0, 1.0]
            self._total_arc = 1.0

    @property
    def total_arc_length(self) -> float:
        return self._total_arc if self._total_arc > 0 else 1.0

    def normalise(self):
        """Scale arc lengths to [0, 1]."""
        if self._total_arc > 1e-12:
            self._samples_arc = [a / self._total_arc for a in self._samples_arc]
        self._total_arc = 1.0

        xs = np.array(self._samples_x)
        arcs = np.array(self._samples_arc)

        # Remove any duplicate x values
        mask = np.concatenate([[True], np.diff(arcs) > 0])
        xs = xs[mask]
        arcs = arcs[mask]

        if len(xs) < 2:
            xs = np.array([0.0, 1.0])
            arcs = np.array([0.0, 1.0])

        self._arc_to_x = sp_interp.interp1d(
            arcs, xs, kind='linear', fill_value='extrapolate'
        )
        self._x_to_arc = sp_interp.interp1d(
            xs, arcs, kind='linear', fill_value='extrapolate'
        )

    def arc_len_to_x(self, arc: float) -> float:
        if self._arc_to_x is None:
            return arc
        return float(self._arc_to_x(arc))

    def x_to_arc_len(self, x: float) -> float:
        if self._x_to_arc is None:
            return x
        return float(self._x_to_arc(x))


def _coupled_polylines(
    dir1: np.ndarray, dir2: np.ndarray,
    pln2img: np.ndarray, img2pln: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Walk two directrix polylines simultaneously, yielding corresponding
    pairs of points and their plane-X.  (Simplified port of
    CoupledPolylinesIterator.)
    """
    pts1, pts2, pxs = [], [], []

    # Sample both directrices in plane-X order
    all_samples = []
    for pt in dir1:
        pln = _apply_homography(img2pln, pt)
        all_samples.append((pln[0], 1, pt.copy()))
    for pt in dir2:
        pln = _apply_homography(img2pln, pt)
        all_samples.append((pln[0], 2, pt.copy()))

    all_samples.sort(key=lambda s: s[0])

    # For each sample, find the corresponding point on the other directrix
    for pln_x, which, img_pt in all_samples:
        gen_top = _apply_homography(pln2img, np.array([pln_x, 0.0]))
        gen_bot = _apply_homography(pln2img, np.array([pln_x, 1.0]))
        gen_dir = gen_bot - gen_top

        if which == 1:
            p1 = img_pt
            p2, _ = _intersect_line_polyline(gen_top, gen_dir, dir2)
        else:
            p2 = img_pt
            p1, _ = _intersect_line_polyline(gen_top, gen_dir, dir1)

        pts1.append(p1)
        pts2.append(p2)
        pxs.append(pln_x)

    return pts1, pts2, pxs


# ===================================================================
#  CylindricalSurfaceDewarper
# ===================================================================

class CylindricalSurfaceDewarper:
    """
    Python port of ScanTailor's CylindricalSurfaceDewarper.

    Given two directrix polylines (top curve and bottom curve of the text
    area), this class builds the mappings needed to dewarp the page.

    Coordinate systems
    ------------------
    img : pixel coords in the warped source image
    pln : plane coords where the 4 directrix endpoints → (0,0),(1,0),(0,1),(1,1)
    crv : dewarped normalised coords (arc-length X, homographic Y)
    """

    def __init__(self, directrix_top: np.ndarray, directrix_bottom: np.ndarray,
                 depth_perception: float = 2.0):
        """
        Parameters
        ----------
        directrix_top    : (N, 2)  top text-boundary polyline (left→right)
        directrix_bottom : (M, 2)  bottom text-boundary polyline (left→right)
        depth_perception : float   camera-distance heuristic  ∈ [1, 3]
        """
        self.dir_top = np.asarray(directrix_top, dtype=np.float64)
        self.dir_bot = np.asarray(directrix_bottom, dtype=np.float64)
        self._depth = depth_perception

        # --- 4-point homography: plane ↔ image -----------------------
        src_corners = np.array([
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [1.0, 1.0],
        ])
        dst_corners = np.array([
            self.dir_top[0], self.dir_top[-1],
            self.dir_bot[0], self.dir_bot[-1],
        ])
        self._pln2img = _four_point_homography_2d(src_corners, dst_corners)
        self._img2pln = _four_point_homography_2d(dst_corners, src_corners)

        # --- straight-line Y in plane coords -------------------------
        self._pln_straight_y = self._calc_straight_line_y()

        # --- arc-length mapper ---------------------------------------
        self._arc_mapper = _ArcLengthMapper()
        self._arc_mapper.build(
            self._depth, self._pln2img, self._img2pln,
            self.dir_top, self.dir_bot,
        )
        self._directrix_arc_length = self._arc_mapper.total_arc_length
        self._arc_mapper.normalise()

    @property
    def directrix_arc_length(self) -> float:
        return self._directrix_arc_length

    # -----------------------------------------------------------------
    #  Core mapping:  mapGeneratrix
    # -----------------------------------------------------------------

    def map_generatrix(self, crv_x: float):
        """
        For a given dewarped column *crv_x* ∈ [0, 1], compute the
        corresponding image-space generatrix line and the 1-D homographic
        mapping that converts dewarped-Y to image-projection along that line.

        Returns
        -------
        gen_p1  : (2,) image-space top point of the generatrix
        gen_p2  : (2,) image-space bottom point of the generatrix
        coeffs  : (3,) 1-D homography coefficients  [a, b, c]
        """
        pln_x = self._arc_mapper.arc_len_to_x(crv_x)

        # Image-space endpoints of the generatrix (from the homography)
        img_top = _apply_homography(self._pln2img, np.array([pln_x, 0.0]))
        img_bot = _apply_homography(self._pln2img, np.array([pln_x, 1.0]))
        gen_dir = img_bot - img_top
        gen_len_sq = np.dot(gen_dir, gen_dir)
        if gen_len_sq < 1e-12:
            return img_top, img_bot, np.array([1.0, 0.0, 0.0])

        # Intersect the generatrix with both directrices
        dir1_pt, _ = _intersect_line_polyline(img_top, gen_dir, self.dir_top)
        dir2_pt, _ = _intersect_line_polyline(img_top, gen_dir, self.dir_bot)

        # Projection scalars along the generatrix
        proj1 = _project_onto_line(img_top, img_bot, dir1_pt)
        proj2 = _project_onto_line(img_top, img_bot, dir2_pt)

        # Straight-line projection
        img_straight = _apply_homography(
            self._pln2img, np.array([pln_x, self._pln_straight_y])
        )
        proj_straight = _project_onto_line(img_top, img_bot, img_straight)

        # Build 3-point 1-D homography:
        #   crv_y=0 → proj1,  crv_y=1 → proj2,  crv_y=straight_y → proj_straight
        if abs(self._pln_straight_y) < 0.05 or abs(self._pln_straight_y - 1.0) < 0.05:
            pairs = np.array([
                [0.0, proj1],
                [1.0, proj2],
                [0.5, 0.5 * (proj1 + proj2)],
            ])
        else:
            pairs = np.array([
                [0.0, proj1],
                [1.0, proj2],
                [self._pln_straight_y, proj_straight],
            ])

        coeffs = _three_point_1d_homography(pairs)
        return img_top, img_bot, coeffs

    # -----------------------------------------------------------------
    #  Bidirectional point mapping
    # -----------------------------------------------------------------

    def map_to_dewarped(self, img_pt: np.ndarray) -> np.ndarray:
        """Map a single image-space point to dewarped normalised coords."""
        pln_pt = _apply_homography(self._img2pln, img_pt)
        pln_x = pln_pt[0]
        crv_x = self._arc_mapper.x_to_arc_len(pln_x)

        img_top = _apply_homography(self._pln2img, np.array([pln_x, 0.0]))
        img_bot = _apply_homography(self._pln2img, np.array([pln_x, 1.0]))
        gen_dir = img_bot - img_top

        dir1_pt, _ = _intersect_line_polyline(img_top, gen_dir, self.dir_top)
        dir2_pt, _ = _intersect_line_polyline(img_top, gen_dir, self.dir_bot)
        img_straight = _apply_homography(
            self._pln2img, np.array([pln_x, self._pln_straight_y])
        )

        proj1 = _project_onto_line(img_top, img_bot, dir1_pt)
        proj2 = _project_onto_line(img_top, img_bot, dir2_pt)
        proj_straight = _project_onto_line(img_top, img_bot, img_straight)
        proj_pt = _project_onto_line(img_top, img_bot, img_pt)

        # Inverse 1-D homography
        if abs(self._pln_straight_y) < 0.05 or abs(self._pln_straight_y - 1.0) < 0.05:
            pairs = np.array([
                [proj1, 0.0],
                [proj2, 1.0],
                [0.5 * (proj1 + proj2), 0.5],
            ])
        else:
            pairs = np.array([
                [proj1, 0.0],
                [proj2, 1.0],
                [proj_straight, self._pln_straight_y],
            ])
        coeffs = _three_point_1d_homography(pairs)
        crv_y = _apply_1d_homography(coeffs, proj_pt)

        return np.array([crv_x, crv_y])

    def map_to_warped(self, crv_pt: np.ndarray) -> np.ndarray:
        """Map a dewarped normalised point back to image-space."""
        gen_top, gen_bot, coeffs = self.map_generatrix(crv_pt[0])
        t = _apply_1d_homography(coeffs, crv_pt[1])
        return gen_top + t * (gen_bot - gen_top)

    # -----------------------------------------------------------------
    #  Private
    # -----------------------------------------------------------------

    def _calc_straight_line_y(self) -> float:
        """
        Find the plane-Y where a horizontal line would appear straight
        in image space when projected through the homography.
        """
        pts1, pts2, pln_xs = _coupled_polylines(
            self.dir_top, self.dir_bot, self._pln2img, self._img2pln
        )

        pln_y_accum = 0.0
        weight_accum = 0.0

        for i in range(len(pln_xs)):
            px = pln_xs[i]
            p1 = pts1[i]
            p2 = pts2[i]
            gen_dir = p2 - p1
            gen_len_sq = np.dot(gen_dir, gen_dir)
            if gen_len_sq < 1e-12:
                continue

            img_line1 = _apply_homography(self._pln2img, np.array([px, 0.0]))
            img_line2 = _apply_homography(self._pln2img, np.array([px, 1.0]))

            proj_p1 = _project_onto_line(p1, p2, img_line1)
            proj_p2 = _project_onto_line(p1, p2, img_line2)

            dp1 = proj_p1 - 0.0
            dp2 = 1.0 - proj_p2
            weight = abs(dp1 + dp2)
            if weight < 0.01:
                continue

            p0 = (proj_p2 * dp1 + proj_p1 * dp2) / (dp1 + dp2)
            img_pt = p1 + p0 * (p2 - p1)
            pln_pt = _apply_homography(self._img2pln, img_pt)
            pln_y_accum += pln_pt[1] * weight
            weight_accum += weight

        return pln_y_accum / weight_accum if weight_accum > 0 else 0.5


# ===================================================================
#  Raster Dewarping – build mesh grid and cv2.remap
# ===================================================================

def _build_dewarp_mesh(
    dewarper: CylindricalSurfaceDewarper,
    dst_w: int, dst_h: int,
    model_domain: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build map_x, map_y arrays for ``cv2.remap``.

    For every pixel (dst_x, dst_y) in the output image, compute the
    corresponding source-image coordinate by:
      1.  Convert dst_x → crv_x via the model domain
      2.  Fetch the generatrix and 1-D homography for that column
      3.  Convert dst_y → crv_y via the model domain
      4.  Apply the 1-D homography to get projScalar along the generatrix
      5.  Read off the image-space (src_x, src_y)

    Parameters
    ----------
    dewarper     : CylindricalSurfaceDewarper
    dst_w, dst_h : output image dimensions
    model_domain : (left, top, right, bottom) in output-pixel coords

    Returns
    -------
    map_x, map_y : float32 arrays of shape (dst_h, dst_w)
    """
    md_left, md_top, md_right, md_bottom = model_domain
    md_w = md_right - md_left
    md_h = md_bottom - md_top

    map_x = np.empty((dst_h, dst_w), dtype=np.float32)
    map_y = np.empty((dst_h, dst_w), dtype=np.float32)

    # Pre-compute crv_y for all rows
    crv_ys = np.zeros(dst_h, dtype=np.float64)
    for y in range(dst_h):
        crv_ys[y] = (y - md_top) / md_h if md_h > 0 else 0.0

    # Column-wise: for each dst_x compute the generatrix once
    for dst_x in range(dst_w):
        crv_x = (dst_x - md_left) / md_w if md_w > 0 else 0.0
        gen_top, gen_bot, coeffs = dewarper.map_generatrix(crv_x)
        vec = gen_bot - gen_top

        for dst_y in range(dst_h):
            t = _apply_1d_homography(coeffs, crv_ys[dst_y])
            src_pt = gen_top + t * vec
            map_x[dst_y, dst_x] = src_pt[0]
            map_y[dst_y, dst_x] = src_pt[1]

    return map_x, map_y


def _compute_model_domain(
    dewarper: CylindricalSurfaceDewarper,
    dir_top: np.ndarray, dir_bot: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute the output-space bounding box (model domain).

    We map a set of points along both directrices into dewarped space,
    then find the bounding box and scale by the directrix arc-length
    ratio to account for stretching compensation.
    """
    # Map directrix endpoints to dewarped space to get bounding region
    crv_pts = []
    for pt in dir_top:
        crv_pts.append(dewarper.map_to_dewarped(pt))
    for pt in dir_bot:
        crv_pts.append(dewarper.map_to_dewarped(pt))
    crv_pts = np.array(crv_pts)

    # Bounding box in dewarped normalised coords should be roughly [0,1]×[0,1]
    # but due to curve extension it may go wider.
    # We need to know what image dimensions we're targeting.
    # Use the original directrix bounding box as baseline.
    all_pts = np.vstack([dir_top, dir_bot])
    left  = float(np.min(all_pts[:, 0]))
    right = float(np.max(all_pts[:, 0]))
    top   = float(np.min(all_pts[:, 1]))
    bottom = float(np.max(all_pts[:, 1]))

    # Scale vertically by the arc-length ratio
    vert_scale = 1.0 / dewarper.directrix_arc_length if dewarper.directrix_arc_length > 0 else 1.0
    center_y = (top + bottom) / 2.0
    new_h = (bottom - top) * vert_scale
    top = center_y - new_h / 2.0
    bottom = center_y + new_h / 2.0

    return (left, top, right, bottom)


# ===================================================================
#  Text-Line Detection  (simplified port of TextLineTracer)
# ===================================================================

def _detect_text_lines(image: np.ndarray) -> List[np.ndarray]:
    """
    Detect text-line polylines in the document image.

    Uses a simplified approach inspired by ScanTailor's text-line tracer:
    1. Binarise (adaptive threshold)
    2. Horizontal morphological closing to merge characters into line blobs
    3. Extract contour centroids / skeletons as polylines
    4. Filter by length and angle

    Returns a list of polylines, each (N, 2) ndarray ordered left→right.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # --- Downscale large images for speed ---
    scale = 1.0
    MAX_DIM = 1200
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
        h, w = gray.shape

    # --- Binarise ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    # --- Horizontal closing to merge text into line-shaped blobs ---
    kern_w = max(w // 8, 40)
    kern_h = max(3, h // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_w, kern_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- Remove small noise ---
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # --- Find contours of the merged line blobs ---
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_length = w * 0.25  # line must span ≥25% of image width

    polylines = []
    for cnt in contours:
        # Bounding rect
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < min_length:
            continue
        if bh > h * 0.15:
            continue  # too tall – probably not a single text line
        if bw < bh * 2:
            continue  # too square

        # Build a polyline from the contour's mid-line (top + bottom averaged)
        # Simplified: sample the contour at N evenly spaced x positions
        n_samples = max(10, bw // 20)
        xs = np.linspace(bx, bx + bw, n_samples)

        # For each x, find the centroid-Y of contour points in that column slice
        cnt_pts = cnt.reshape(-1, 2).astype(np.float64)
        ys = []
        valid_xs = []
        for x in xs:
            # Points within ±2 pixels of this x
            mask = np.abs(cnt_pts[:, 0] - x) < max(3, bw / n_samples)
            if mask.sum() > 0:
                y_min = cnt_pts[mask, 1].min()
                y_max = cnt_pts[mask, 1].max()
                ys.append((y_min + y_max) / 2.0)
                valid_xs.append(x)

        if len(valid_xs) < 4:
            continue

        pts = np.column_stack([valid_xs, ys])

        # Smooth the polyline
        if len(pts) > 5:
            from scipy.ndimage import uniform_filter1d
            pts[:, 1] = uniform_filter1d(pts[:, 1], size=3)

        # Scale back to original image coordinates
        pts /= scale

        polylines.append(pts)

    # Sort polylines top-to-bottom by their average Y
    polylines.sort(key=lambda p: np.mean(p[:, 1]))

    return polylines


# ===================================================================
#  DistortionModelBuilder  (RANSAC top/bottom curve selection)
# ===================================================================

def _select_directrices(
    polylines: List[np.ndarray], img_w: int, img_h: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    From a list of detected text-line polylines, select the best TOP and
    BOTTOM directrix curves.

    Strategy (inspired by ScanTailor's DistortionModelBuilder):
    1. Score each pair of (top, bottom) curves by how well the other curves
       become straight horizontal lines when dewarped.
    2. As a fast heuristic, try the top-3 and bottom-3 curves, plus some
       random pairs.

    Returns two polylines (top, bottom) or (None, None) if no good pair found.
    """
    n = len(polylines)
    if n < 2:
        return None, None

    # Ensure all polylines are left→right and extend across much of image width
    valid = []
    for p in polylines:
        if p[-1, 0] < p[0, 0]:
            p = p[::-1]
        span = p[-1, 0] - p[0, 0]
        if span > img_w * 0.2:
            valid.append(p)

    if len(valid) < 2:
        return None, None

    # Sort by average Y
    valid.sort(key=lambda p: np.mean(p[:, 1]))

    # Extend curves to image edges using linear extrapolation
    extended = []
    for p in valid:
        ep = _extend_polyline_to_bounds(p, img_w)
        extended.append(ep)

    n = len(extended)

    # Try combinations: top-3 × bottom-3 + random
    best_pair = None
    best_score = np.inf

    candidates = []
    for i in range(min(3, n)):
        for j in range(max(0, n - 3), n):
            if i < j:
                candidates.append((i, j))

    # Add some random pairs
    rng = np.random.RandomState(42)
    for _ in range(min(10, n * n)):
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i > j:
            i, j = j, i
        if i < j and (i, j) not in candidates:
            candidates.append((i, j))

    for i, j in candidates:
        top_c = extended[i]
        bot_c = extended[j]

        # Quick validation: the quadrilateral must be convex
        quad = np.array([top_c[0], top_c[-1], bot_c[-1], bot_c[0]])
        if not _is_convex_quad(quad):
            continue

        # Score: build a dewarper and measure how straight the middle curves become
        score = _score_model(top_c, bot_c, extended)
        if score < best_score:
            best_score = score
            best_pair = (top_c, bot_c)

    if best_pair is None and len(extended) >= 2:
        # Fallback: just use first and last
        best_pair = (extended[0], extended[-1])

    return best_pair


def _extend_polyline_to_bounds(polyline: np.ndarray, img_w: int) -> np.ndarray:
    """Extend a polyline leftwards to x=0 and rightwards to x=img_w-1."""
    pts = list(polyline)

    # Extend left
    if pts[0][0] > 5:
        # Linear extrapolation from first two points
        d = pts[1] - pts[0]
        if abs(d[0]) > 1e-6:
            slope = d[1] / d[0]
            y_at_0 = pts[0][1] - slope * pts[0][0]
            pts.insert(0, np.array([0.0, y_at_0]))

    # Extend right
    if pts[-1][0] < img_w - 5:
        d = pts[-1] - pts[-2]
        if abs(d[0]) > 1e-6:
            slope = d[1] / d[0]
            y_at_end = pts[-1][1] + slope * (img_w - 1 - pts[-1][0])
            pts.append(np.array([img_w - 1.0, y_at_end]))

    return np.array(pts, dtype=np.float64)


def _is_convex_quad(quad: np.ndarray) -> bool:
    """Check if 4 points form a convex quadrilateral."""
    n = 4
    sign = None
    for i in range(n):
        p0 = quad[i]
        p1 = quad[(i + 1) % n]
        p2 = quad[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - \
                (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) < 1e-6:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def _score_model(
    top: np.ndarray, bot: np.ndarray,
    all_curves: List[np.ndarray],
) -> float:
    """
    Score a (top, bottom) directrix pair by how straight the other curves
    become after dewarping.  Lower is better.
    """
    try:
        dewarper = CylindricalSurfaceDewarper(top, bot, depth_perception=2.0)
    except Exception:
        return 1e12

    total_error = 0.0
    for curve in all_curves:
        # Map a few sample points from this curve to dewarped space
        n_samples = min(len(curve), 15)
        indices = np.linspace(0, len(curve) - 1, n_samples, dtype=int)
        dewarped_pts = []
        for idx in indices:
            try:
                dp = dewarper.map_to_dewarped(curve[idx])
                if np.isfinite(dp).all():
                    dewarped_pts.append(dp)
            except Exception:
                pass

        if len(dewarped_pts) < 3:
            total_error += 10.0
            continue

        dewarped_pts = np.array(dewarped_pts)

        # Fit a line y = ax + b and measure residuals
        xs = dewarped_pts[:, 0]
        ys = dewarped_pts[:, 1]
        if np.ptp(xs) < 1e-6:
            total_error += 10.0
            continue

        try:
            coeffs = np.polyfit(xs, ys, 1)
            residuals = ys - np.polyval(coeffs, xs)
            total_error += float(np.mean(np.abs(residuals)))
        except Exception:
            total_error += 10.0

    return total_error


# ===================================================================
#  Public API
# ===================================================================

def apply_grid_dewarp(
    image: np.ndarray,
    strength: float = 1.0,
    depth_perception: float = 2.0,
) -> np.ndarray:
    """
    ScanTailor-Advanced-style cylindrical-surface mesh dewarping.

    Detects text-line curves, selects top/bottom directrices, builds a
    cylindrical surface model, and remaps the image.

    Parameters
    ----------
    image            : Input document image (BGR or grayscale).
    strength         : Correction strength multiplier (1.0 = full).
    depth_perception : Camera distance heuristic (1.0–3.0, default 2.0).

    Returns
    -------
    Dewarped image – same shape and dtype as input.
    """
    h, w = image.shape[:2]

    # 1. Detect text lines
    polylines = _detect_text_lines(image)
    print(f"[MeshDewarp] Detected {len(polylines)} text-line polylines")

    if len(polylines) < 2:
        print("[MeshDewarp] Too few text lines — skipping dewarp.")
        return image.copy()

    # 2. Select top and bottom directrix curves
    dir_top, dir_bot = _select_directrices(polylines, w, h)
    if dir_top is None or dir_bot is None:
        print("[MeshDewarp] Could not find valid directrix pair — skipping.")
        return image.copy()

    # 3. Build dewarper
    try:
        dewarper = CylindricalSurfaceDewarper(dir_top, dir_bot, depth_perception)
    except Exception as e:
        print(f"[MeshDewarp] Dewarper construction failed: {e}")
        return image.copy()

    # 4. Compute model domain
    md = _compute_model_domain(dewarper, dir_top, dir_bot)

    # 5. Build mesh grid
    map_x, map_y = _build_dewarp_mesh(dewarper, w, h, md)

    # Apply strength: blend between identity map and dewarped map
    if abs(strength - 1.0) > 0.01:
        id_x, id_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = id_x + strength * (map_x - id_x)
        map_y = id_y + strength * (map_y - id_y)

    # 6. Remap
    dewarped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    print(f"[MeshDewarp] Applied cylindrical-surface mesh dewarp "
          f"(strength={strength:.2f}, depth={depth_perception:.1f})")
    return dewarped


def apply_custom_grid_dewarp(
    image: np.ndarray,
    row_curves: List[List[List[float]]],
    strength: float = 1.0,
    depth_perception: float = 2.0,
) -> np.ndarray:
    """
    Apply dewarping using a manually-edited grid from the frontend.

    Parameters
    ----------
    image      : Input document image.
    row_curves : List of curves; each curve is a list of [x, y] control points.
                 Must be at least 2 curves (top boundary + bottom boundary).
    strength   : Correction strength multiplier (1.0 = full).
    depth_perception : Camera distance heuristic (1.0–3.0).

    Returns
    -------
    Dewarped image.
    """
    h, w = image.shape[:2]
    R = len(row_curves)

    if R < 2:
        print("[MeshDewarp] Too few row curves in custom grid — need ≥2.")
        return image.copy()

    # ── Smooth all row curves using bilinear-consistent averaging ──────
    # Matches ScanTailor's RasterDewarper approach: simple linear
    # interpolation smoothing that removes local spikes while
    # preserving the overall curve shape.
    smoothed_curves = []
    for curve_pts in row_curves:
        pts = np.array(curve_pts, dtype=np.float64)
        if len(pts) < 4:
            smoothed_curves.append(pts)
            continue

        # If it's a sparse curve from the frontend (e.g., 7 control points),
        # evaluate a dense cubic spline to get a perfectly smooth directrix.
        if len(pts) < 20:
            from scipy.interpolate import CubicSpline
            try:
                # Parametrize by chord length
                distances = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
                t = np.concatenate(([0], np.cumsum(distances)))
                cs = CubicSpline(t, pts, bc_type='natural')
                t_dense = np.linspace(0, t[-1], 100)
                pts = cs(t_dense)
            except Exception as e:
                print(f"[MeshDewarp] Spline eval failed: {e}")
        else:
            # Dense polyline: uniform moving-average filter (bilinear-consistent smoothing)
            from scipy.ndimage import uniform_filter1d
            window = min(5, len(pts) // 2)  # window size
            if window >= 2:
                # Two passes for smoother result
                pts[:, 0] = uniform_filter1d(pts[:, 0], size=window, mode='nearest')
                pts[:, 1] = uniform_filter1d(pts[:, 1], size=window, mode='nearest')
                pts[:, 0] = uniform_filter1d(pts[:, 0], size=window, mode='nearest')
                pts[:, 1] = uniform_filter1d(pts[:, 1], size=window, mode='nearest')

        smoothed_curves.append(pts)

    # Use the first and last smoothed curves as directrices
    dir_top_pts = smoothed_curves[0]
    dir_bot_pts = smoothed_curves[-1]

    # Ensure left→right ordering
    if dir_top_pts[-1, 0] < dir_top_pts[0, 0]:
        dir_top_pts = dir_top_pts[::-1]
    if dir_bot_pts[-1, 0] < dir_bot_pts[0, 0]:
        dir_bot_pts = dir_bot_pts[::-1]

    # Build dewarper
    try:
        dewarper = CylindricalSurfaceDewarper(dir_top_pts, dir_bot_pts, depth_perception)
    except Exception as e:
        print(f"[MeshDewarp] Custom dewarper failed: {e}")
        return image.copy()

    # Model domain
    md = _compute_model_domain(dewarper, dir_top_pts, dir_bot_pts)

    # Build mesh
    map_x, map_y = _build_dewarp_mesh(dewarper, w, h, md)

    # Strength blending
    if abs(strength - 1.0) > 0.01:
        id_x, id_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = id_x + strength * (map_x - id_x)
        map_y = id_y + strength * (map_y - id_y)

    # Remap
    dewarped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    print(f"[MeshDewarp] Applied custom mesh dewarp: "
          f"{R} curves, strength={strength:.2f}")
    return dewarped



def analyze_dewarp_grid(
    image: np.ndarray,
    n_cols: int = 20,
) -> dict:
    """
    Analyse the document's warp mesh WITHOUT applying dewarping.

    Returns the detected directrix curves and a mesh grid of (x, y) points
    that ScanTailor renders as the blue control-point overlay.

    Parameters
    ----------
    image  : Input document image.
    n_cols : Number of vertical grid lines (columns in the mesh).

    Returns
    -------
    dict with keys:
        detected   : bool
        width, height : int
        row_curves : [[[x,y], ...], ...]   polylines for top, detected rows, bottom
        col_lines  : [[[x,y], ...], ...]   vertical connectors
        row_count  : int                   number of interior text rows
    """
    h, w = image.shape[:2]

    polylines = _detect_text_lines(image)
    if len(polylines) < 2:
        return {"detected": False, "width": w, "height": h,
                "row_curves": [], "col_lines": [], "row_count": 0}

    dir_top, dir_bot = _select_directrices(polylines, w, h)
    if dir_top is None or dir_bot is None:
        return {"detected": False, "width": w, "height": h,
                "row_curves": [], "col_lines": [], "row_count": 0}

    # Build dewarper to get the mesh
    try:
        dewarper = CylindricalSurfaceDewarper(dir_top, dir_bot, depth_perception=2.0)
    except Exception:
        return {"detected": False, "width": w, "height": h,
                "row_curves": [], "col_lines": [], "row_count": 0}

    # Sample crv_x at n_cols positions
    crv_xs = np.linspace(0, 1, n_cols)
    # Dense row sampling to match ScanTailor's ~30-40 horizontal grid lines
    n_rows = max(30, n_cols)
    crv_ys = np.linspace(0, 1, n_rows)

    # Build the mesh grid by mapping (crv_x, crv_y) → image space
    grid = np.zeros((n_rows, n_cols, 2), dtype=np.float64)
    for ci, cx in enumerate(crv_xs):
        gen_top, gen_bot, coeffs = dewarper.map_generatrix(cx)
        vec = gen_bot - gen_top
        for ri, cy in enumerate(crv_ys):
            t = _apply_1d_homography(coeffs, cy)
            pt = gen_top + t * vec
            grid[ri, ci] = pt

    # Build row_curves (each row of the mesh)
    row_curves = []
    for ri in range(n_rows):
        row_curves.append([[float(grid[ri, ci, 0]), float(grid[ri, ci, 1])]
                           for ci in range(n_cols)])

    # Build col_lines (each column of the mesh)
    col_lines = []
    for ci in range(n_cols):
        col_lines.append([[float(grid[ri, ci, 0]), float(grid[ri, ci, 1])]
                          for ri in range(n_rows)])

    # ── Sparse control points for the frontend spline editor ──────────
    # 7 evenly-spaced handles extracted from detected directrix curves.
    # The frontend renders smooth Catmull-Rom splines through these handles.
    N_CP = 7
    top_idxs = np.linspace(0, len(dir_top) - 1, N_CP, dtype=int)
    bot_idxs = np.linspace(0, len(dir_bot) - 1, N_CP, dtype=int)
    top_cps = [[float(dir_top[i][0]), float(dir_top[i][1])] for i in top_idxs]
    bot_cps = [[float(dir_bot[i][0]), float(dir_bot[i][1])] for i in bot_idxs]

    return {
        "detected": True,
        "width": w,
        "height": h,
        "row_curves": row_curves,
        "col_lines": col_lines,
        "row_count": len(polylines),
        "topCPs": top_cps,
        "botCPs": bot_cps,
    }
