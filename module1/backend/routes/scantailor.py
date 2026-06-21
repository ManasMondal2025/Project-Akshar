"""
Module 1 — ScanTailor Tools API
================================
New stateless endpoints for the 5 ScanTailor-style document filters:

  POST /orient/rotate      — Rotate image 90 / -90 / 180 degrees
  POST /orient/auto        — Auto-detect best orientation
  POST /split/detect       — Detect layout type + split position
  POST /split/apply        — Split image into left + right pages
  POST /content/detect     — Auto-detect content box boundary
  POST /content/apply      — Crop image to detected content area
  POST /margins/detect     — Auto-detect page content boundary
  POST /margins/apply      — Crop + pad with specified margins
"""

import os
import uuid
import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from utils.page_split import detect_page_layout, apply_split, split_page_geometry, detect_spine_x, split_at
from utils.margins import detect_margins, apply_margins
from utils.content_selection import detect_content_box_api, apply_content_selection

router = APIRouter()

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")
    return img


def _save_encode(img: np.ndarray, prefix: str = "processed") -> dict:
    uid = str(uuid.uuid4())
    out_path = os.path.join(PROCESSED_DIR, f"{prefix}_{uid}.png")
    cv2.imwrite(out_path, img)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode()
    h, w = img.shape[:2]
    return {
        "image_path": out_path,
        "preview": f"data:image/png;base64,{b64}",
        "width": w,
        "height": h,
    }


# ── Request Models ────────────────────────────────────────────────────────────

class RotateRequest(BaseModel):
    image_path: str
    angle: int = 90          # 90 = CW, -90 = CCW, 180 = flip


class AutoOrientRequest(BaseModel):
    image_path: str


class SplitDetectRequest(BaseModel):
    image_path: str


class SplitApplyRequest(BaseModel):
    image_path: str
    split_x: float              # pixel x position of the split line (fallback)
    layout_type: str = "two_pages"
    content_x1: int = 0        # left content boundary
    content_x2: Optional[int] = None
    selected_side: str = "both"         # 'left' | 'right' | 'both'
    boundary: Optional[list] = None
    page_layout: Optional[dict] = None  # Full PageLayout dict — preferred path


class MarginsDetectRequest(BaseModel):
    image_path: str
    dpi: int = 300


class MarginsApplyRequest(BaseModel):
    image_path: str
    top_mm: float    = 5.0
    bottom_mm: float = 5.0
    left_mm: float   = 10.0
    right_mm: float  = 10.0
    dpi: int         = 300


class ContentDetectRequest(BaseModel):
    image_path: str
    dpi: int = 300


class ContentApplyRequest(BaseModel):
    image_path: str
    content_rect: Optional[list[int]] = None
    dpi: int = 300


# ── Fix Orientation ───────────────────────────────────────────────────────────

@router.post("/orient/rotate")
async def rotate_image(req: RotateRequest):
    """
    Rotate the image by exactly 90°, −90°, or 180°.

    angle:  90  → clockwise 90°
           -90  → counter-clockwise 90°
           180  → upside-down
    """
    img = _load(req.image_path)

    angle = req.angle
    # Normalise to 90 / -90 / 180
    angle = int(angle) % 360
    if angle == 270:
        angle = -90

    if angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle in (-90, 270):
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    else:
        raise HTTPException(status_code=400, detail="angle must be 90, -90, or 180")

    result = _save_encode(rotated, "oriented")
    return {
        "status": "success",
        "angle_applied": req.angle,
        "message": f"Rotated {req.angle}°",
        **result,
    }


@router.post("/orient/auto")
async def auto_orient(req: AutoOrientRequest):
    """
    Heuristically suggest the best orientation (portrait vs. landscape).
    Returns suggested_angle = 0 (no change) or 90 (rotate CW to portrait).
    """
    img = _load(req.image_path)
    h, w = img.shape[:2]

    suggested_angle = 90 if w > h * 1.2 else 0

    return {
        "status": "success",
        "suggested_angle": suggested_angle,
        "current_width": w,
        "current_height": h,
        "is_landscape": bool(w > h),
    }


# ── Split Pages ───────────────────────────────────────────────────────────────

@router.post("/split/detect")
async def split_detect(req: SplitDetectRequest):
    """
    Detect whether the image contains one page or two pages.
    Uses the full ScanTailor two-pass pipeline (VertLineFinder + ContentSpanFinder).
    Returns layout_type, split_x, split_x_ratio, confidence, offcut flags.
    """
    img = _load(req.image_path)

    try:
        result = detect_page_layout(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layout detection failed: {exc}")

    return {"status": "success", **result}


@router.post("/split/detect-spine")
async def split_detect_spine(req: SplitDetectRequest):
    """
    Fast spine detection using projection-profile dip + Hough-line vote.

    This is the 'Auto-Detect Spine' option (Option 1) in the frontend Split
    Pages panel. Returns the detected spine x-coordinate and immediately
    encoded left/right pixel-crop previews — no second /split/apply call needed.

    Returns:
      detected      : bool
      spine_x       : int | null   — pixel x of the spine
      spine_x_ratio : float | null — spine_x / image_width
      left_page     : { image_path, preview, width, height } | null
      right_page    : { image_path, preview, width, height } | null
    """
    img = _load(req.image_path)
    h, w = img.shape[:2]

    try:
        spine_x = detect_spine_x(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spine detection failed: {exc}")

    if spine_x is None:
        return {
            "status": "success",
            "detected": False,
            "spine_x": None,
            "spine_x_ratio": None,
            "left_page": None,
            "right_page": None,
        }

    left, right = split_at(img, spine_x)
    return {
        "status": "success",
        "detected": True,
        "spine_x": spine_x,
        "spine_x_ratio": round(spine_x / w, 4),
        "left_page":  _save_encode(left,  "spine_left"),
        "right_page": _save_encode(right, "spine_right"),
    }


@router.post("/split/apply")
async def split_apply(req: SplitApplyRequest):
    """
    Split (or crop) the image using detected layout.

    PRIMARY path: if page_layout is provided, uses geometry-first split:
        PageLayout → left/right polygon → perspective warp → image

    FALLBACK: pixel-crop at split_x if page_layout absent.
    """
    img = _load(req.image_path)
    h, w = img.shape[:2]
    content_x2 = req.content_x2 if req.content_x2 is not None else w

    try:
        # ── Geometry-first path (preferred) ──────────────────────────────────────
        if req.page_layout:
            crops = split_page_geometry(
                img,
                page_layout_dict = req.page_layout,
                selected_side    = req.selected_side,
            )
        else:
            # ── Legacy pixel-crop fallback ──────────────────────────────────────
            cx1Px = req.content_x1
            cx2Px = content_x2
            spxPx = int(np.clip(int(req.split_x), 0, w))
            crops = apply_split(
                img,
                split_x      = spxPx,
                layout_type  = req.layout_type,
                content_x1   = cx1Px,
                content_x2   = cx2Px,
                selected_side= req.selected_side,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Split failed: {exc}")

    def _crop_to_result(key: str):
        """Decode base64 crop → save to disk → return _save_encode dict."""
        if key not in crops or not crops[key]:
            return None
        import io
        raw = base64.b64decode(crops[key])
        buf = np.frombuffer(raw, np.uint8)
        img_dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_dec is None:
            return None
        return _save_encode(img_dec, f'split_{key}')

    response = {
        'status':      'success',
        'layout_type': req.layout_type,
        'split_x':     req.split_x,
        'used_geometry': bool(req.page_layout),
    }

    if req.layout_type == 'two_pages':
        response.update(
            left_page  = _crop_to_result('left'),
            right_page = _crop_to_result('right'),
        )
    else:
        response.update(
            left_page  = _crop_to_result('page'),
            right_page = None,
        )

    return response


# ── Select Content ────────────────────────────────────────────────────────────

@router.post("/content/detect")
async def content_detect(req: ContentDetectRequest):
    """
    Auto-detect the content box boundary using ScanTailor's
    Wolf binarization + max-whitespace + trim pipeline.

    Returns content_box {x, y, width, height} and content_rect [x1, y1, x2, y2].
    """
    img = _load(req.image_path)

    try:
        result = detect_content_box_api(img, dpi=req.dpi)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Content detection failed: {exc}")

    annotated = result.pop("annotated_image", None)
    if annotated is not None:
        saved = _save_encode(annotated, "content_detect")
        result["preview"] = saved["preview"]
        result["temp_image_path"] = saved["image_path"]

    return {"status": "success", **result}


@router.post("/content/apply")
async def content_apply(req: ContentApplyRequest):
    """
    Crop image to the detected or manually adjusted content area.

    If req.content_rect is provided [x1, y1, x2, y2], crops to that area.
    Otherwise, auto-detects and crops using the Select Content algorithm.
    """
    img = _load(req.image_path)
    h, w = img.shape[:2]

    try:
        if req.content_rect:
            x1, y1, x2, y2 = req.content_rect
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(w, int(round(x2)))
            y2 = min(h, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                result_img = img.copy()
            else:
                result_img = img[y1:y2, x1:x2].copy()
        else:
            result_img = apply_content_selection(img, dpi=req.dpi)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Content selection failed: {exc}")

    result = _save_encode(result_img, "content")
    return {
        "status": "success",
        "message": "Content selection applied",
        **result,
    }


# ── Margins ───────────────────────────────────────────────────────────────────

@router.post("/margins/detect")
async def margins_detect(req: MarginsDetectRequest):
    """
    Auto-detect page content boundaries.

    Returns top/bottom/left/right in both pixels and millimetres,
    plus content_rect [x1, y1, x2, y2].
    """
    img = _load(req.image_path)

    try:
        result = detect_margins(img, dpi=req.dpi)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Margin detection failed: {exc}")

    return {"status": "success", **result}


@router.post("/margins/apply")
async def margins_apply(req: MarginsApplyRequest):
    """
    Crop image to content, then add white border padding for the given margins.

    Accepts margins in millimetres; dpi used for unit conversion (default 150).
    """
    img = _load(req.image_path)

    try:
        result_img = apply_margins(
            img,
            top_mm=req.top_mm,
            bottom_mm=req.bottom_mm,
            left_mm=req.left_mm,
            right_mm=req.right_mm,
            dpi=req.dpi,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Margin application failed: {exc}")

    result = _save_encode(result_img, "margins")
    return {
        "status": "success",
        "margins_applied": {
            "top_mm": req.top_mm, "bottom_mm": req.bottom_mm,
            "left_mm": req.left_mm, "right_mm": req.right_mm,
        },
        **result,
    }
