from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from utils.poly_dewarp.curve_model import Curve, Point


def evaluate_curve(curve: Curve, x_values: np.ndarray) -> np.ndarray:
    """Evaluate an editable cubic B-spline across arbitrary image columns."""
    control_x = np.asarray([point.x for point in curve.control_points], dtype=np.float64)
    control_y = np.asarray([point.y for point in curve.control_points], dtype=np.float64)
    order = np.argsort(control_x)
    control_x = control_x[order]
    control_y = control_y[order]
    control_x, unique_indexes = np.unique(control_x, return_index=True)
    control_y = control_y[unique_indexes]
    degree = min(3, len(control_x) - 1)
    if degree < 1:
        return np.full_like(x_values, control_y[0] if len(control_y) else 0)
    spline = make_interp_spline(control_x, control_y, k=degree)
    return spline(np.clip(x_values, control_x[0], control_x[-1]))


def sample_spline(curve: Curve, width: int, sample_count: int = 160) -> list[Point]:
    x_values = np.linspace(0, width - 1, sample_count)
    y_values = evaluate_curve(curve, x_values)
    return [Point(x=float(x), y=float(y)) for x, y in zip(x_values, y_values)]


def spline_length(curve: Curve, width: int) -> float:
    samples = sample_spline(curve, width, sample_count=max(160, width // 4))
    values = np.asarray([(point.x, point.y) for point in samples])
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def spline_rmse(curve: Curve) -> float:
    if not curve.sample_points:
        return 0.0
    x_values = np.asarray([point.x for point in curve.sample_points])
    observed_y = np.asarray([point.y for point in curve.sample_points])
    fitted_y = evaluate_curve(curve, x_values)
    return float(np.sqrt(np.mean((fitted_y - observed_y) ** 2)))


def refresh_curve_metrics(curve: Curve, width: int) -> Curve:
    curve.spline_points = sample_spline(curve, width)
    curve.spline_length = spline_length(curve, width)
    curve.rmse = spline_rmse(curve)
    return curve

