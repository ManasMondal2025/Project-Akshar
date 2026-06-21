from __future__ import annotations

import cv2
import numpy as np

MAX_PREVIEW_DIMENSION = 1000


def make_preview(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a responsive editing image and its scale relative to the original."""
    height, width = image.shape[:2]
    scale = min(1.0, MAX_PREVIEW_DIMENSION / max(height, width))
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def binary_text_image(image: np.ndarray) -> np.ndarray:
    """Create a foreground-white mask tuned for dark printed text on light paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        13,
    )
    # Remove tiny camera noise while leaving character strokes intact.
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

