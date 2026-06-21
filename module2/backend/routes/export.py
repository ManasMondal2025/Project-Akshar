"""
PROJECT AKSHAR - PDF Export Routes (Module 2)
=============================================
Moved from Module 1 — served on port 8001.

  POST /export/pdf          — plain PDF from ordered images
  POST /export/pdf-bbox     — PDF with colour-coded layout bounding boxes
                              rendered from the SAME PDF that OCR ran on,
                              so bbox pixel coordinates map exactly.
"""

import os
import uuid
from typing import List

import fitz  # PyMuPDF
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from utils.pdf_utils import images_to_pdf

router = APIRouter()

# Shared processed/ directory lives in module1/backend/processed/
# (same location module2's dewarp routes write to)
_M1_BACKEND = os.path.join(
    os.path.dirname(__file__),          # module2/backend/routes/
    "..", "..", "..",                   # project root
    "module1", "backend", "processed",
)
PROCESSED_DIR = os.path.normpath(_M1_BACKEND)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    """Request model for plain PDF export."""
    image_paths: List[str]
    filename: str = "document_output.pdf"


class OcrBlock(BaseModel):
    """One layout block from the module-3 JSON output."""
    page: int           # 1-indexed page number
    type: str = "other" # title | section_header | paragraph | list | table | figure | other
    bbox: List[float]   # [x1, y1, x2, y2] in pixels at *ocr_dpi* resolution
    text: str = ""


class ExportBBoxRequest(BaseModel):
    """Request model for PDF export with bounding-box overlay."""
    pdf_path: str            # server-side path to the PDF that OCR ran on
    blocks: List[OcrBlock]   # layout blocks from module-3
    filename: str = "annotated_layout.pdf"
    ocr_dpi: float = 300.0   # DPI the OCR pipeline used when rendering pages


# ---------------------------------------------------------------------------
# Colour palette per block type  (fill alpha, stroke RGB)
# ---------------------------------------------------------------------------

_FILL = {
    "title":          (0.12, 0.47, 0.94, 0.18),
    "section_header": (0.29, 0.69, 0.93, 0.18),
    "paragraph":      (0.18, 0.80, 0.44, 0.18),
    "list":           (0.60, 0.35, 0.95, 0.18),
    "table":          (0.97, 0.58, 0.02, 0.22),
    "figure":         (0.95, 0.26, 0.21, 0.15),
    "other":          (0.55, 0.55, 0.55, 0.12),
}
_STROKE = {
    "title":          (0.12, 0.47, 0.94),
    "section_header": (0.29, 0.69, 0.93),
    "paragraph":      (0.18, 0.80, 0.44),
    "list":           (0.60, 0.35, 0.95),
    "table":          (0.97, 0.58, 0.02),
    "figure":         (0.95, 0.26, 0.21),
    "other":          (0.55, 0.55, 0.55),
}
_DEFAULT_FILL   = (0.55, 0.55, 0.55, 0.12)
_DEFAULT_STROKE = (0.55, 0.55, 0.55)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/export/pdf")
async def export_pdf(request: ExportRequest):
    """
    Convert one or more processed images into a downloadable PDF.
    """
    if not request.image_paths:
        raise HTTPException(status_code=400, detail="No image paths provided")

    for path in request.image_paths:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    output_id = str(uuid.uuid4())
    output_pdf_path = os.path.join(PROCESSED_DIR, f"export_{output_id}.pdf")

    try:
        images_to_pdf(request.image_paths, output_pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

    with open(output_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{request.filename}"'},
    )


@router.post("/export/pdf-bbox")
async def export_pdf_bbox(request: ExportBBoxRequest):
    """
    Render the original PDF at the same DPI OCR used, then overlay
    colour-coded bounding boxes from the layout-parser JSON blocks.

    Block-type → colour mapping:
      title          → blue
      section_header → light-blue
      paragraph      → green
      list           → purple
      table          → orange
      figure         → red
      other          → grey
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {request.pdf_path}")

    blocks_by_page: dict = {}
    for blk in request.blocks:
        blocks_by_page.setdefault(blk.page, []).append(blk)

    pts_per_px = 72.0 / request.ocr_dpi

    output_id = str(uuid.uuid4())
    output_pdf_path = os.path.join(PROCESSED_DIR, f"bbox_export_{output_id}.pdf")

    try:
        src_doc = fitz.open(request.pdf_path)
        out_doc = fitz.open()

        zoom = request.ocr_dpi / 72.0
        mat  = fitz.Matrix(zoom, zoom)

        for page_idx in range(len(src_doc)):
            page_num = page_idx + 1

            src_page = src_doc[page_idx]
            pix = src_page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")

            w_pt = pix.width  * pts_per_px
            h_pt = pix.height * pts_per_px

            out_page = out_doc.new_page(width=w_pt, height=h_pt)
            out_page.insert_image(fitz.Rect(0, 0, w_pt, h_pt), stream=img_bytes)

            for blk in blocks_by_page.get(page_num, []):
                if len(blk.bbox) < 4:
                    continue

                x1 = blk.bbox[0] * pts_per_px
                y1 = blk.bbox[1] * pts_per_px
                x2 = blk.bbox[2] * pts_per_px
                y2 = blk.bbox[3] * pts_per_px

                rect = fitz.Rect(x1, y1, x2, y2)

                fr, fg, fb, fa = _FILL.get(blk.type, _DEFAULT_FILL)
                sr, sg, sb     = _STROKE.get(blk.type, _DEFAULT_STROKE)

                out_page.draw_rect(
                    rect,
                    color=(sr, sg, sb),
                    fill=(fr, fg, fb),
                    fill_opacity=fa,
                    width=1.0,
                )

                label = blk.type.replace("_", " ").title()
                out_page.insert_text(
                    fitz.Point(x1 + 2, y1 + 7),
                    label,
                    fontsize=5.5,
                    color=(sr, sg, sb),
                )

        src_doc.close()
        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        out_doc.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BBox PDF export failed: {str(e)}")

    with open(output_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{request.filename}"'},
    )
