"""
PROJECT AKSHAR - Dewarp + Deskew Routes
"""

import os
import uuid
import base64
import cv2
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from utils.scantailor import apply_grid_dewarp, analyze_dewarp_grid, apply_custom_grid_dewarp
from utils.dewarp_ml.predictor import run_auto_dewarp, is_model_available
from utils.deskew import deskew_image, apply_manual_deskew
from utils.poly_dewarp.runner import run_poly_dewarp

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


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DewarpRequest(BaseModel):
    """Request model for cylindrical-surface mesh dewarping."""
    image_path: str
    strength: float = 1.0  # 0.5 to 1.5; 1.0 = full correction
    depth_perception: float = 2.0  # camera distance heuristic (1.0–3.0)
    row_curves: Optional[List[List[List[float]]]] = None  # manual grid from frontend


class AutoDewarpRequest(BaseModel):
    """Request model for ML-based automatic dewarping (ICCV 2023 model)."""
    image_path: str


class DeskewRequest(BaseModel):
    """Request model for auto deskew."""
    image_path: str


class ManualDeskewRequest(BaseModel):
    """Request model for manual deskew with user-specified angle."""
    image_path: str
    angle: float  # degrees; positive = CW, negative = CCW


class AnalyzeGridRequest(BaseModel):
    """Request model for grid analysis visualization."""
    image_path: str
    n_cols: int = 20  # number of vertical grid lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_and_encode(image, prefix: str = "processed") -> dict:
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


def _load_image(image_path: str):
    """Load and validate an image from the given path."""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    return img


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/dewarp")
async def dewarp_image(request: DewarpRequest):
    """
    Apply ScanTailor-Advanced-style cylindrical-surface mesh dewarping.

    Detects text-line curves to build top/bottom directrix polylines,
    constructs a cylindrical surface model (4-point homography + arc-length
    mapping + generatrix 1-D homography), then remaps with bicubic
    interpolation.

    Args:
        image_path:       Path to source image.
        strength:         Correction strength (default 1.0 = full).
        depth_perception: Camera distance heuristic (1.0–3.0, default 2.0).
        row_curves:       Optional manual grid from the frontend editor.
    """
    img = _load_image(request.image_path)

    try:
        if request.row_curves:
            dewarped = apply_custom_grid_dewarp(
                img, request.row_curves,
                strength=request.strength,
                depth_perception=request.depth_perception,
            )
        else:
            dewarped = apply_grid_dewarp(
                img,
                strength=request.strength,
                depth_perception=request.depth_perception,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh dewarp failed: {str(e)}")

    result = _save_and_encode(dewarped, "dewarped")
    return {
        "status": "success",
        "method": "mesh_dewarp",
        "strength": request.strength,
        "depth_perception": request.depth_perception,
        "message": f"ScanTailor cylindrical mesh dewarp applied "
                   f"(strength={request.strength:.2f}, depth={request.depth_perception:.1f})",
        **result
    }


@router.post("/deskew")
async def auto_deskew(request: DeskewRequest):
    """
    Auto-detect skew angle from the document and correct it.

    Uses the automatic deskew logic ported from deskew_img.py:
    projection-profile scoring first, Hough fallback second, min-area fallback
    last, then rotates without cutting the page canvas.
    Returns the detected angle along with the corrected image.
    """
    img = _load_image(request.image_path)

    try:
        deskewed, angle = deskew_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deskew failed: {str(e)}")

    result = _save_and_encode(deskewed, "deskewed")
    return {
        "status": "success",
        "method": "auto_deskew",
        "detected_angle": round(angle, 3),
        "message": f"Auto deskew applied (detected angle: {angle:.2f}°)",
        **result
    }


@router.post("/deskew/manual")
async def manual_deskew(request: ManualDeskewRequest):
    """
    Apply a user-specified rotation correction.

    Args:
        image_path: Path to source image.
        angle:      Rotation angle in degrees. Positive = CW, negative = CCW.
    """
    if not -90 <= request.angle <= 90:
        raise HTTPException(
            status_code=400,
            detail="Angle must be between -90 and 90 degrees"
        )

    img = _load_image(request.image_path)

    try:
        deskewed = apply_manual_deskew(img, angle_deg=request.angle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual deskew failed: {str(e)}")

    result = _save_and_encode(deskewed, "deskewed_manual")
    return {
        "status": "success",
        "method": "manual_deskew",
        "angle_applied": request.angle,
        "message": f"Manual deskew applied (angle: {request.angle:.2f}°)",
        **result
    }


@router.post("/dewarp/auto")
async def auto_dewarp_ml(request: AutoDewarpRequest):
    """
    Apply ML-based automatic document dewarping using the ICCV 2023 neural network
    (Foreground and Text-lines Aware Document Image Rectification).

    The model runs fully automatically — no grid interaction required.
    On first call the model is lazy-loaded (~1-2 s); subsequent calls are fast.

    Args:
        image_path: Path to source image.
    """
    if not is_model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Auto-dewarp model not available. "
                "Please ensure '30.pt' exists in the test dewarp pretrained_models/ folder, "
                "or set the DEWARP_MODEL_PATH environment variable."
            )
        )

    img = _load_image(request.image_path)

    try:
        dewarped = run_auto_dewarp(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML dewarp failed: {str(e)}")

    result = _save_and_encode(dewarped, "auto_dewarped")
    return {
        "status": "success",
        "method": "ml_dewarp",
        "message": "ICCV 2023 neural network auto-dewarp applied",
        **result,
    }


@router.post("/dewarp/analyze-grid")
async def analyze_grid(request: AnalyzeGridRequest):
    """
    Analyze the document warp mesh WITHOUT applying any correction.
    Returns the detected row-curve grid as sampled (x, y) point arrays —
    same data ScanTailor uses to render its blue grid overlay.

    Use this BEFORE applying /dewarp to preview the detected grid on the canvas.

    Returns:
      detected: bool — whether a usable grid was found
      row_curves: list of polylines (one per text row + top/bottom borders)
      col_lines: list of polylines (vertical connectors at evenly-spaced x positions)
      row_count: number of text rows detected
      width, height: image dimensions (for coordinate scaling)
    """
    img = _load_image(request.image_path)

    try:
        grid = analyze_dewarp_grid(img, n_cols=request.n_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid analysis failed: {str(e)}")

    return {
        "status": "success",
        **grid,
    }


# ---------------------------------------------------------------------------
# Poly Dewarp (B-spline, ported from dewarp2)
# ---------------------------------------------------------------------------

class PolyDewarpRequest(BaseModel):
    """Request model for B-spline poly dewarping (from dewarp2 logic)."""
    image_path: str


class PolyEstimateRequest(BaseModel):
    """Request model for estimating poly dewarp curves (visualization only)."""
    image_path: str


@router.post("/dewarp/poly/estimate-curves")
async def poly_estimate_curves(request: PolyEstimateRequest):
    """
    Estimate the 4 B-spline text-density curves WITHOUT applying dewarping.

    Returns the 4 curves as sampled polylines for canvas visualization.
    The frontend can overlay these on the image so the user can see
    the detected curves before applying poly dewarp.

    Returns:
      detected: bool
      curves: list of 4 polylines, each a list of {x, y} points
      width, height: image dimensions
    """
    img = _load_image(request.image_path)
    height, width = img.shape[:2]

    try:
        from utils.poly_dewarp.curve_estimation import estimate_four_curves
        curves = estimate_four_curves(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Poly curve estimation failed: {str(e)}")

    # Convert Curve objects to plain polyline dicts for the frontend
    curve_polylines = []
    for curve in curves:
        pts = [{"x": p.x, "y": p.y} for p in curve.spline_points]
        cps = [{"x": p.x, "y": p.y} for p in curve.control_points]
        curve_polylines.append({
            "id": curve.id,
            "name": curve.name,
            "color": curve.color,
            "points": pts,
            "control_points": cps,
        })

    return {
        "status": "success",
        "detected": True,
        "curves": curve_polylines,
        "width": width,
        "height": height,
    }


class PolyDewarpRequest(BaseModel):
    """Request model for B-spline poly dewarping (from dewarp2 logic)."""
    image_path: str
    custom_curves: Optional[List[dict]] = None  # updated control points from frontend


@router.post("/dewarp/poly")
async def poly_dewarp(request: PolyDewarpRequest):
    """
    Apply B-spline poly dewarping using the dewarp2 algorithm.

    If custom_curves is provided (list of {id, name, color, control_points}),
    uses those control points for the deformation (user-adjusted).
    Otherwise runs the automatic estimation.
    """
    img = _load_image(request.image_path)

    try:
        if request.custom_curves and len(request.custom_curves) >= 4 and all("control_points" in c for c in request.custom_curves):
            from utils.poly_dewarp.curve_model import Curve as PolyCurve, Point as PolyPoint
            from utils.poly_dewarp.spline_fit import refresh_curve_metrics
            from utils.poly_dewarp.deformation import remap
            custom_curves = []
            for c in request.custom_curves:
                curve = PolyCurve(
                    id=c["id"],
                    name=c["name"],
                    color=c["color"],
                    control_points=[PolyPoint(x=p["x"], y=p["y"]) for p in c["control_points"]],
                )
                curve = refresh_curve_metrics(curve, img.shape[1])
                custom_curves.append(curve)
            dewarped = remap(img, custom_curves)
        else:
            dewarped = run_poly_dewarp(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Poly dewarp failed: {str(e)}")

    result = _save_and_encode(dewarped, "poly_dewarped")
    return {
        "status": "success",
        "method": "poly_dewarp",
        "message": "B-spline poly dewarp applied (dewarp2 algorithm)",
        **result,
    }
