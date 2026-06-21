"""
Module 1 - Corner Detection Route
Handles automatic corner detection and perspective warp from detected corners.
"""

import os
import uuid
import base64
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from utils.edge_detect_corners import detect_corners, warp_from_corners
from utils.cnn_corner_detect import detect_corners_cnn

router = APIRouter()

# Directory for processed files
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


class DetectRequest(BaseModel):
    """Request model for corner detection."""
    image_path: str
    method: str = "classical"  # "classical" | "cnn"


class ApplyRequest(BaseModel):
    """Request model for corner warp application."""
    image_path: str
    corners: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


def save_and_encode(image, prefix: str = "corners") -> dict:
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


@router.post("/corners/detect")
async def detect_corners_endpoint(request: DetectRequest):
    """
    Detect the four document corners in an image.

    Input:
        - image_path: path to the source image

    Returns: detected corner coordinates [TL, TR, BR, BL] and image dimensions.
    """
    # Validate image exists
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Read the image
    img = cv2.imread(request.image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    h, w = img.shape[:2]

    try:
        if request.method == "cnn":
            corners = detect_corners_cnn(request.image_path)
        else:
            corners = detect_corners(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corner detection failed: {str(e)}")

    return {
        "status": "success",
        "corners": corners,
        "width": w,
        "height": h,
    }


@router.post("/corners/apply")
async def apply_corners_endpoint(request: ApplyRequest):
    """
    Apply a perspective warp using 4 corner coordinates.

    Input:
        - image_path: path to the source image
        - corners: 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns: warped image path and base64 preview.
    """
    # Validate image exists
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Validate corner count
    if len(request.corners) != 4:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 4 corners, got {len(request.corners)}"
        )

    # Read the image
    img = cv2.imread(request.image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    try:
        warped = warp_from_corners(img, request.corners)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corner warp failed: {str(e)}")

    result = save_and_encode(warped, "corners")
    return {
        "status": "success",
        "message": "Corner warp applied",
        **result
    }
