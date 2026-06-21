from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from utils.poly_dewarp.preprocessing import binary_text_image
from utils.poly_dewarp.spline_fit import refresh_curve_metrics
from utils.poly_dewarp.text_region import detect_text_region
from utils.poly_dewarp.curve_model import Curve, Point

CURVE_RATIOS = (0.125, 0.375, 0.500, 0.625, 0.875)
CURVE_COLORS = ("#ef4444", "#22c55e", "#f97316", "#3b82f6", "#eab308")
STRIP_COUNT = 20
CONTROL_POINT_COUNT = 7


def _candidate_density_rows(
    profile: np.ndarray,
    target_y: float,
    band_radius: int,
    region_top: int,
    region_bottom: int,
) -> list[tuple[float, float]]:
    low = max(region_top, round(target_y - band_radius))
    high = min(region_bottom, round(target_y + band_radius))
    if high <= low:
        return [(target_y, 0.0)]
    band = gaussian_filter1d(profile[low : high + 1].astype(float), sigma=2.0)
    strongest = float(band.max())
    if strongest <= 0:
        return [(target_y, 0.0)]
    peaks, _ = find_peaks(band, distance=max(2, band_radius // 4))
    if not peaks.size:
        peaks = np.asarray([int(np.argmax(band))])
    return [
        (float(low + peak), float(band[peak] / strongest))
        for peak in peaks
    ]


def _track_density_rows(
    profiles: list[np.ndarray],
    target_y: float,
    band_radius: int,
    region_top: int,
    region_bottom: int,
    region_height: int,
) -> list[float]:
    """Choose a strong path without jumping between nearby text lines."""
    candidates = [
        _candidate_density_rows(profile, target_y, band_radius, region_top, region_bottom)
        for profile in profiles
    ]
    transition_scale = max(5.0, region_height * 0.012)
    costs = [
        -3.0 * strength + 0.25 * abs(row - target_y) / max(1, band_radius)
        for row, strength in candidates[0]
    ]
    parents: list[list[int]] = []

    for strip_candidates in candidates[1:]:
        previous_candidates = candidates[len(parents)]
        next_costs: list[float] = []
        next_parents: list[int] = []
        for row, strength in strip_candidates:
            local_cost = -3.0 * strength + 0.25 * abs(row - target_y) / max(1, band_radius)
            options = [
                cost + 0.85 * abs(row - previous_row) / transition_scale
                for cost, (previous_row, _) in zip(costs, previous_candidates)
            ]
            best_parent = int(np.argmin(options))
            next_costs.append(local_cost + options[best_parent])
            next_parents.append(best_parent)
        costs = next_costs
        parents.append(next_parents)

    candidate_index = int(np.argmin(costs))
    rows = [candidates[-1][candidate_index][0]]
    for strip_index in range(len(parents) - 1, -1, -1):
        candidate_index = parents[strip_index][candidate_index]
        rows.append(candidates[strip_index][candidate_index][0])
    return list(reversed(rows))


def _smooth_samples(points: list[Point]) -> list[Point]:
    y_values = np.asarray([point.y for point in points])
    smoothed_y = gaussian_filter1d(y_values, sigma=1.15, mode="nearest")
    return [
        Point(x=point.x, y=float(y))
        for point, y in zip(points, smoothed_y)
    ]


def _control_points_from_samples(samples: list[Point], width: int) -> list[Point]:
    sample_x = np.asarray([point.x for point in samples])
    sample_y = np.asarray([point.y for point in samples])
    control_x = np.linspace(0, width - 1, CONTROL_POINT_COUNT)
    control_y = np.interp(control_x, sample_x, sample_y)
    return [Point(x=float(x), y=float(y)) for x, y in zip(control_x, control_y)]


def estimate_curves(image: np.ndarray) -> list[Curve]:
    """Estimate evenly distributed text-density curves (one per CURVE_RATIOS entry) without OCR."""
    binary = binary_text_image(image)
    height, width = binary.shape
    _, region_top, _, region_bottom = detect_text_region(binary)
    region_height = max(1, region_bottom - region_top)
    band_radius = max(6, round(region_height * 0.095))
    edges = np.linspace(0, width, STRIP_COUNT + 1, dtype=int)
    targets = [region_top + ratio * region_height for ratio in CURVE_RATIOS]
    profiles: list[np.ndarray] = []
    strip_centers: list[float] = []
    for strip_index in range(STRIP_COUNT):
        left, right = edges[strip_index], max(edges[strip_index] + 1, edges[strip_index + 1])
        profiles.append(binary[:, left:right].sum(axis=1) / 255.0)
        strip_centers.append(float((left + right - 1) / 2))

    tracked_rows = [
        _track_density_rows(
            profiles,
            target,
            band_radius,
            region_top,
            region_bottom,
            region_height,
        )
        for target in targets
    ]
    points_by_curve: list[list[Point]] = [[] for _ in CURVE_RATIOS]
    minimum_spacing = max(4.0, region_height * 0.13)
    for strip_index, x in enumerate(strip_centers):
        rows = [curve_rows[strip_index] for curve_rows in tracked_rows]
        for index in range(1, len(rows)):
            rows[index] = max(rows[index], rows[index - 1] + minimum_spacing)
        overflow = max(0.0, rows[-1] - region_bottom)
        rows = [row - overflow for row in rows]
        for index, row in enumerate(rows):
            points_by_curve[index].append(Point(x=x, y=float(row)))

    curves: list[Curve] = []
    for index, rough_points in enumerate(points_by_curve):
        samples = _smooth_samples(rough_points)
        curve = Curve(
            id=f"curve-{index + 1}",
            name=f"Curve {index + 1}",
            color=CURVE_COLORS[index],
            control_points=_control_points_from_samples(samples, width),
            sample_points=samples,
        )
        curves.append(refresh_curve_metrics(curve, width))
    return curves


# Backwards-compatible alias so existing callers (runner.py, dewarp.py) need no changes
estimate_four_curves = estimate_curves
