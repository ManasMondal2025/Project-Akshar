"""
Poly Dewarp — public API entry point.
Ported from dewarp2 with local imports.
"""
from __future__ import annotations

import cv2
import numpy as np

from utils.poly_dewarp.curve_estimation import estimate_four_curves
from utils.poly_dewarp.deformation import remap


def run_poly_dewarp(image: np.ndarray) -> np.ndarray:
    """
    Apply B-spline poly dewarping to a document image.

    Automatically estimates 4 text-density curves (top, upper-mid, lower-mid, bottom)
    and applies an interpolation-based deformation to flatten the page.

    Args:
        image: BGR image as numpy array.

    Returns:
        Dewarped BGR image as numpy array.
    """
    curves = estimate_four_curves(image)
    return remap(image, curves)
