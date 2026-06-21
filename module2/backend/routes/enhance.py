"""
PROJECT AKSHAR - Enhancement Routes
Provides endpoints for Otsu threshold and Adaptive threshold enhancement.
Each endpoint accepts an optional output_format parameter:
  - "color"     → keep original color channels
  - "grayscale" → convert to single-channel grayscale (3-channel output for consistency)
  - "bw"        → pure black & white (already output by Otsu/Adaptive)
"""

import os
import uuid
import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from utils.enhance import apply_otsu_threshold, apply_adaptive_threshold

router = APIRouter()

# Directory for processed files
# Write output images to module1's shared processed/ directory
_M1_BACKEND = os.path.join(
    os.path.dirname(__file__),          # module2/backend/routes/
    "..", "..", "..",                   # project root
    "module1", "backend", "processed",
)
PROCESSED_DIR = os.path.normpath(_M1_BACKEND)
os.makedirs(PROCESSED_DIR, exist_ok=True)


class EnhanceRequest(BaseModel):
    """Request model for image enhancement endpoints."""
    image_path: str
    output_format: str = "bw"  # "color" | "grayscale" | "bw"


def _apply_output_format(image: np.ndarray, original: np.ndarray, fmt: str, method: str) -> np.ndarray:
    """
    Apply output format conversion after enhancement.

    Args:
        image:    The enhanced (already thresholded) image.
        original: The original pre-enhancement BGR image.
        fmt:      One of "color", "grayscale", "bw".
        method:   Enhancement method name for logging.

    Returns:
        Formatted image as 3-channel BGR.
    """
    fmt = fmt.lower()

    if fmt == "bw":
        # Already binary — ensure 3-channel
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    elif fmt == "grayscale":
        # Convert enhanced result to gray, then back to BGR for consistency
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif fmt == "color":
        # For color: use the enhanced binary as a mask to darken the original color image
        # Convert enhanced to single-channel mask
        if len(image.shape) == 3:
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            mask = image
        # Normalize mask to [0, 1] float
        mask_f = mask.astype(np.float32) / 255.0
        # Apply mask to original color channels
        result = (original.astype(np.float32) * mask_f[:, :, np.newaxis]).astype(np.uint8)
        return result

    else:
        # Unknown format — return as-is
        return image


def save_and_encode(image: np.ndarray, prefix: str) -> dict:
    """Save a processed image and return its path + base64 encoding."""
    output_id = str(uuid.uuid4())
    output_path = os.path.join(PROCESSED_DIR, f"{prefix}_{output_id}.png")
    cv2.imwrite(output_path, image)

    _, buffer = cv2.imencode('.png', image)
    b64 = base64.b64encode(buffer).decode('utf-8')
    height, width = image.shape[:2]

    return {
        "image_path": output_path,
        "preview": f"data:image/png;base64,{b64}",
        "width": width,
        "height": height
    }


def load_image(image_path: str) -> np.ndarray:
    """Load and validate an image from the given path."""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    return img


@router.post("/enhance/otsu")
async def enhance_otsu(request: EnhanceRequest):
    """
    Apply the workbench Otsu/BW enhancement.
    The UI name is kept, but this endpoint now returns Module 1-style visual
    enhancement rather than forcing pure binary output.
    Supports output_format: 'color', 'grayscale', 'bw'.
    """
    original = load_image(request.image_path)

    try:
        # Determine the mode dynamically based on the requested output format ('color' or 'bw')
        mode = "color" if request.output_format == "color" else "bw"
        result_img = apply_otsu_threshold(original, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otsu threshold failed: {str(e)}")

    result = save_and_encode(result_img, "otsu")
    return {
        "status": "success",
        "method": "otsu",
        "output_format": request.output_format,
        "message": f"Otsu/BW enhancement applied ({request.output_format})",
        **result
    }


@router.post("/enhance/adaptive")
async def enhance_adaptive(request: EnhanceRequest):
    """
    Apply adaptive Gaussian thresholding.
    Better for documents with varying lighting conditions.
    Supports output_format: 'color', 'grayscale', 'bw'.
    """
    original = load_image(request.image_path)

    try:
        enhanced = apply_adaptive_threshold(original)
        result_img = _apply_output_format(enhanced, original, request.output_format, "adaptive")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive threshold failed: {str(e)}")

    result = save_and_encode(result_img, "adaptive")
    return {
        "status": "success",
        "method": "adaptive",
        "output_format": request.output_format,
        "message": f"Adaptive Gaussian threshold applied ({request.output_format})",
        **result
    }
