"""
Module 1 - Transform Route
Handles perspective transformation requests with 4 corner coordinates.
"""

import os
import uuid
import base64
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

from utils.transform import apply_perspective_transform

router = APIRouter()

# Directory for processed files
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


class TransformRequest(BaseModel):
    """Request model for perspective transform."""
    image_path: str
    corners: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


def save_and_encode(image, prefix: str = "transformed") -> dict:
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


@router.post("/transform")
async def transform_image(request: TransformRequest):
    """
    Apply perspective transformation to rectify a document.
    
    Input:
        - image_path: path to the source image
        - corners: 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns: transformed image path and base64 preview
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
    
    # Convert corners to tuples
    corners = [(c[0], c[1]) for c in request.corners]
    
    try:
        # Apply perspective transform
        warped = apply_perspective_transform(img, corners)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")
    
    result = save_and_encode(warped, "transformed")
    return {
        "status": "success",
        "message": "Perspective transform applied",
        **result
    }
