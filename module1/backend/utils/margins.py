"""
Module 1 — ScanTailor-Style Margin Detection & Application
===========================================================
Mirrors ScanTailor's Margins filter:
  1. detect_margins()  — Otsu binarise → largest contour → bounding rect → margins in px & mm
  2. apply_margins()   — crop to content, then pad with white border

Unit conversion uses the DPI stored with (or assumed for) the image.
Default assumed DPI is 300 dpi (matching the PDF render DPI).
"""

import cv2
import numpy as np
from typing import Dict, Any


# ── Unit helpers ──────────────────────────────────────────────────────────────

def mm_to_px(mm: float, dpi: int = 300) -> int:
    """Convert millimetres → pixels at given DPI."""
    return max(0, int(round(mm * dpi / 25.4)))


def px_to_mm(px: int, dpi: int = 300) -> float:
    """Convert pixels → millimetres at given DPI."""
    return round(px * 25.4 / dpi, 2)


# ── Margin Detection ──────────────────────────────────────────────────────────

def detect_margins(image: np.ndarray, dpi: int = 300) -> Dict[str, Any]:
    """
    Detect document content boundaries by finding the largest dark contour.

    Returns
    -------
    dict with keys:
      top_px, bottom_px, left_px, right_px  — margins in pixels
      top_mm, bottom_mm, left_mm, right_mm  — margins in millimetres
      content_rect                            — [x1, y1, x2, y2] pixel rect of content
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Blur + threshold (inverse: content = white)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to fill small gaps in text blocks
    ksize  = max(5, min(31, w // 40))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours (external only)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: 10 px all around
        pad = 10
        return {
            "top_px": pad, "bottom_px": pad, "left_px": pad, "right_px": pad,
            "top_mm": px_to_mm(pad, dpi), "bottom_mm": px_to_mm(pad, dpi),
            "left_mm": px_to_mm(pad, dpi), "right_mm": px_to_mm(pad, dpi),
            "content_rect": [pad, pad, w - pad, h - pad],
        }

    # Largest contour = document body
    largest  = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    top    = max(0, y)
    left   = max(0, x)
    bottom = max(0, h - (y + ch))
    right  = max(0, w - (x + cw))

    return {
        "top_px":    top,    "bottom_px": bottom,
        "left_px":   left,   "right_px":  right,
        "top_mm":    px_to_mm(top, dpi),    "bottom_mm": px_to_mm(bottom, dpi),
        "left_mm":   px_to_mm(left, dpi),   "right_mm":  px_to_mm(right, dpi),
        "content_rect": [x, y, x + cw, y + ch],
    }


# ── Margin Application ────────────────────────────────────────────────────────

def apply_margins(
    image: np.ndarray,
    top_mm: float,
    bottom_mm: float,
    left_mm: float,
    right_mm: float,
    dpi: int = 300,
) -> np.ndarray:
    """
    Add white padding for the given margins to the whole page.
    (Previously cropped to detected content area, now adds margin to the full image).
    """
    t = mm_to_px(top_mm,    dpi)
    b = mm_to_px(bottom_mm, dpi)
    l = mm_to_px(left_mm,   dpi)
    r = mm_to_px(right_mm,  dpi)

    fill = (255, 255, 255) if len(image.shape) == 3 else 255
    result = cv2.copyMakeBorder(
        image, t, b, l, r,
        borderType=cv2.BORDER_CONSTANT,
        value=fill,
    )
    return result
