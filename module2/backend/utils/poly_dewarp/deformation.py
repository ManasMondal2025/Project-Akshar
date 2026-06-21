from __future__ import annotations

import cv2
import numpy as np

from utils.poly_dewarp.spline_fit import evaluate_curve
from utils.poly_dewarp.curve_model import Curve


def _ordered_curve_field(
    curves: list[Curve],
    width: int,
    height: int,
    coordinate_scale: float,
) -> np.ndarray:
    x_values = np.arange(width, dtype=np.float32)
    source_x = x_values / coordinate_scale
    evaluated = np.asarray(
        [evaluate_curve(curve, source_x) * coordinate_scale for curve in curves],
        dtype=np.float32,
    )
    minimum_spacing = min(max(3.0, height * 0.035), (height - 3.0) / (len(curves) + 1))
    evaluated[0] = np.clip(evaluated[0], 1, height - 2)
    for index in range(1, len(evaluated)):
        evaluated[index] = np.maximum(evaluated[index], evaluated[index - 1] + minimum_spacing)
    evaluated[-1] = np.minimum(evaluated[-1], height - 2)
    for index in range(len(evaluated) - 2, -1, -1):
        evaluated[index] = np.minimum(evaluated[index], evaluated[index + 1] - minimum_spacing)
    evaluated[0] = np.maximum(evaluated[0], 1)
    for index in range(1, len(evaluated)):
        evaluated[index] = np.maximum(evaluated[index], evaluated[index - 1] + minimum_spacing)
    return evaluated


def deformation_maps(
    shape: tuple[int, ...],
    curves: list[Curve],
    coordinate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map flat output rows to their corresponding source rows."""
    height, width = shape[:2]
    field = _ordered_curve_field(curves, width, height, coordinate_scale)
    target_rows = field.mean(axis=1)
    target_rows = np.maximum.accumulate(target_rows)
    target_anchors = np.concatenate(([0.0], target_rows, [height - 1.0]))
    output_rows = np.arange(height, dtype=np.float32)

    map_y = np.empty((height, width), dtype=np.float32)
    for x in range(width):
        source_anchors = np.concatenate(([0.0], field[:, x], [height - 1.0]))
        map_y[:, x] = np.interp(output_rows, target_anchors, source_anchors)
    map_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
    return map_x, map_y


def remap(image: np.ndarray, curves: list[Curve], coordinate_scale: float = 1.0) -> np.ndarray:
    map_x, map_y = deformation_maps(image.shape, curves, coordinate_scale)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
