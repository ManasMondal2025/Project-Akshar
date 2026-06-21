"""
Module 1 - Upload Route
Handles image file uploads, saves to disk, and returns file path + base64 preview.
"""

import os
import uuid
import base64
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

# Directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64-encoded string."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for processing.
    
    Accepts: image files (JPEG, PNG, BMP, TIFF)
    Returns: file_id, file_path, base64 preview, and image dimensions
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )
    
    # Generate unique filename
    ext = os.path.splitext(file.filename)[1] or ".png"
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file
    try:
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Read image to get dimensions and generate preview
    img = cv2.imread(filepath)
    if img is None:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    height, width = img.shape[:2]
    preview_base64 = image_to_base64(filepath)
    
    return {
        "file_id": file_id,
        "file_path": filepath,
        "filename": file.filename,
        "width": width,
        "height": height,
        "preview": f"data:image/png;base64,{preview_base64}"
    }
