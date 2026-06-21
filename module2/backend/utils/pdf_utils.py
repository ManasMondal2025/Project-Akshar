"""
Module 1 - PDF Utilities
========================
Handles PDF type detection (digital vs scanned), page extraction as images,
and converting ordered image lists back into a PDF.

Detection Strategy:
  - Open the PDF with PyMuPDF
  - For each page, count extractable text characters
  - If average chars/page > threshold (50), classify as "digital"
  - Otherwise classify as "scanned"
"""

import os
import io
from typing import List, Literal

import fitz  # PyMuPDF
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pages with more average chars than this are considered "digital"
DIGITAL_TEXT_THRESHOLD = 300

# DPI for rendering scanned PDF pages to images
PDF_RENDER_DPI = 300


# ---------------------------------------------------------------------------
# Type Detection
# ---------------------------------------------------------------------------

def detect_pdf_type(pdf_path: str) -> Literal["digital", "scanned"]:
    """
    Determine whether a PDF is a digital (text-based) or scanned (image-based)
    document.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        "digital" if the PDF has extractable text above threshold,
        "scanned" otherwise.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a valid PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    total_chars = 0
    num_pages = len(doc)

    for page in doc:
        text = page.get_text("text")
        total_chars += len(text.strip())

    doc.close()

    if num_pages == 0:
        return "scanned"

    avg_chars = total_chars / num_pages
    result: Literal["digital", "scanned"] = (
        "digital" if avg_chars >= DIGITAL_TEXT_THRESHOLD else "scanned"
    )

    print(
        f"[PDFUtils] '{os.path.basename(pdf_path)}': "
        f"{num_pages} pages, avg {avg_chars:.0f} chars/page → {result}"
    )
    return result


# ---------------------------------------------------------------------------
# Page Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(pdf_path: str, output_dir: str, dpi: int = PDF_RENDER_DPI) -> List[str]:
    """
    Render each page of a PDF into a PNG image.

    Args:
        pdf_path:   Absolute path to the PDF.
        output_dir: Directory where page images will be saved.
        dpi:        Render resolution (default 300 DPI).

    Returns:
        List of absolute paths to the generated PNG images, ordered by page.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths: List[str] = []

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    zoom = dpi / 72.0  # fitz default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for page_idx, page in enumerate(doc):
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(output_dir, f"{base_name}_page_{page_idx + 1:04d}.png")
        pixmap.save(out_path)
        image_paths.append(out_path)
        print(f"[PDFUtils] Extracted page {page_idx + 1} → {out_path}")

    doc.close()
    return image_paths


# ---------------------------------------------------------------------------
# PDF Assembly
# ---------------------------------------------------------------------------

def images_to_pdf(image_paths: List[str], output_path: str) -> str:
    """
    Convert an ordered list of images into a single PDF file using PyMuPDF.

    Args:
        image_paths: Ordered list of absolute paths to image files.
        output_path: Where to save the output PDF.

    Returns:
        The absolute path of the saved PDF.
    """
    if not image_paths:
        raise ValueError("No images provided to convert to PDF.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = fitz.open()  # new empty PDF

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[PDFUtils] Warning: image not found, skipping: {img_path}")
            continue

        # Open image and get dimensions
        with Image.open(img_path) as pil_img:
            w_px, h_px = pil_img.size

        # Convert pixel dimensions to points (72 pt/inch, 96 px/inch assumed)
        # fitz inserts images at full size; use a standard A4-ish rect
        # We scale to fit within A4 (595 × 842 points) preserving aspect ratio
        max_w, max_h = 595, 842
        scale = min(max_w / w_px, max_h / h_px)
        w_pt = int(w_px * scale)
        h_pt = int(h_px * scale)

        page = doc.new_page(width=w_pt, height=h_pt)
        rect = fitz.Rect(0, 0, w_pt, h_pt)
        page.insert_image(rect, filename=img_path)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    print(f"[PDFUtils] Assembled {len(image_paths)} pages → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Digital PDF Text Extraction (for Module 3 bypass)
# ---------------------------------------------------------------------------

def extract_digital_pdf_text(pdf_path: str):
    """
    Extract text blocks with bounding boxes from a digital PDF using PyMuPDF.
    Used when a digital PDF bypasses OCR entirely.

    Returns:
        List of dicts: [{"text": str, "bbox": [x0,y0,x1,y1], "page_num": int}, ...]
    """
    doc = fitz.open(pdf_path)
    blocks_out = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] == 0:  # text block
                text = b[4].strip()
                if text:
                    blocks_out.append({
                        "text": text,
                        "bbox": [b[0], b[1], b[2], b[3]],
                        "page_num": page_idx + 1,
                    })

    doc.close()
    return blocks_out
