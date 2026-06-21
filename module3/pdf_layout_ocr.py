#!/usr/bin/env python3
"""
pdf_layout_ocr.py  (module3)
────────────────────────────
Local layout detection + OCR engine for Module 3.

Uses PaddleOCR's PP-DocLayout_plus-L model (20 semantic classes) to detect
document regions (title, section_header, paragraph, list, table) and
PaddleOCR's PP-OCRv5 engine for text extraction.

NO external API keys required — fully local inference.

Public API (used by sarvam_client.py):
    process_pdf(pdf_path, out_json, layout_filter, dpi, use_gpu, visualize, vis_dir)
        → List[Dict]   (one dict per extracted block)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from typing import List, Dict, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PP-DocLayout label → canonical block type
# ─────────────────────────────────────────────────────────────────────────────

TYPE_TITLE          = "title"
TYPE_SECTION_HEADER = "section_header"
TYPE_PARAGRAPH      = "paragraph"
TYPE_LIST           = "list"
TYPE_TABLE          = "table"
TYPE_OTHER          = "other"

_LABEL_MAP: Dict[str, str] = {
    # titles / headings
    "doc_title":          TYPE_TITLE,
    "paragraph_title":    TYPE_SECTION_HEADER,
    # body text
    "text":               TYPE_PARAGRAPH,
    "content":            TYPE_PARAGRAPH,
    "abstract":           TYPE_PARAGRAPH,
    "aside_text":         TYPE_PARAGRAPH,
    "reference":          TYPE_PARAGRAPH,
    "reference_content":  TYPE_PARAGRAPH,
    # lists
    "number":             TYPE_LIST,
    # tables
    "table":              TYPE_TABLE,
    # structural / decorative
    "header":             TYPE_OTHER,
    "footer":             TYPE_OTHER,
    "footnote":           TYPE_OTHER,
    "algorithm":          TYPE_OTHER,
    "formula":            TYPE_OTHER,
    "formula_number":     TYPE_OTHER,
    "figure_title":       TYPE_OTHER,
}

# Visual-only labels — no text to OCR
_VISUAL_LABELS = {"image", "figure", "seal", "chart"}

# All labels that carry text
_TEXT_BEARING_LABELS = set(_LABEL_MAP.keys())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Model singletons
# ─────────────────────────────────────────────────────────────────────────────

_layout_model = None
_ocr_engine   = None


def _get_layout_model():
    """Load PP-DocLayout_plus-L once and cache for the process lifetime."""
    global _layout_model
    if _layout_model is None:
        try:
            from paddleocr import LayoutDetection  # type: ignore
        except ImportError:
            raise ImportError(
                "paddleocr >= 3.x required.\n"
                "Activate myenv1 and run: pip install paddleocr paddlepaddle"
            )
        print("  [layout] Loading PP-DocLayout_plus-L model…")
        _layout_model = LayoutDetection()
        print("  [layout] Model ready.")
    return _layout_model


def _get_ocr_engine(use_gpu: bool = False):
    """Load PaddleOCR once and cache for the process lifetime."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError:
            raise ImportError(
                "paddleocr >= 3.x required.\n"
                "Activate myenv1 and run: pip install paddleocr paddlepaddle"
            )
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        print("  [ocr] Initialising PaddleOCR engine…")
        device = "gpu" if use_gpu else "cpu"
        try:
            _ocr_engine = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                device=device,
            )
        except TypeError:
            # Older paddleocr API fallback
            _ocr_engine = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=use_gpu,
            )
        print("  [ocr] Engine ready.")
    return _ocr_engine


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Layout detection + heuristic re-classifier
# ─────────────────────────────────────────────────────────────────────────────

class Block:
    """One detected layout block."""

    def __init__(self, label: str, score: float, coord: List[float]):
        self.raw_label = label
        self.score     = score
        self.x1 = float(coord[0])
        self.y1 = float(coord[1])
        self.x2 = float(coord[2])
        self.y2 = float(coord[3])
        self.type: str = _LABEL_MAP.get(label, TYPE_OTHER)

    @property
    def width(self)  -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def area(self)   -> float: return self.width * self.height

    def __repr__(self) -> str:
        return (
            f"Block(type={self.type!r}, raw={self.raw_label!r}, "
            f"score={self.score:.2f}, h={self.height:.0f}, w={self.width:.0f})"
        )


def detect_blocks(image_bgr: np.ndarray) -> List[Block]:
    """Run PP-DocLayout on a BGR image and return detected blocks."""
    model = _get_layout_model()
    results = model.predict(image_bgr)

    blocks: List[Block] = []
    for page_result in results:
        for box in (page_result.get("boxes") or []):
            label = box.get("label", "")
            if label in _VISUAL_LABELS:
                continue
            score = float(box.get("score", 0.0))
            coord = [float(v) for v in box.get("coordinate", [0, 0, 0, 0])]
            if label in _TEXT_BEARING_LABELS:
                blocks.append(Block(label=label, score=score, coord=coord))
    return blocks


def _heuristic_reclassify(blocks: List[Block], page_h: int, page_w: int) -> None:
    """
    Refine block types in-place using document-level geometry heuristics.

    Rules applied in order:
      1. Single-line paragraph spanning >55% of page width → section_header
      2. Block near top 12% of page + above-average height  → promote to title
    """
    if not blocks:
        return

    para_heights = [b.height for b in blocks if b.type == TYPE_PARAGRAPH]
    if not para_heights:
        return
    median_h = float(np.median(para_heights))

    for b in blocks:
        # Rule 1 — single-line wide paragraph → section header
        if b.type == TYPE_PARAGRAPH:
            is_single_line = b.height < 1.6 * median_h
            spans_page     = b.width / page_w > 0.55
            if is_single_line and spans_page and b.height > median_h * 0.85:
                b.type = TYPE_SECTION_HEADER

        # Rule 2 — near top of page → promote to title
        if b.type in (TYPE_PARAGRAPH, TYPE_SECTION_HEADER):
            near_top = b.y1 / page_h < 0.12
            tall_box = b.height > median_h * 1.1
            if near_top and tall_box:
                b.type = TYPE_TITLE


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OCR helpers
# ─────────────────────────────────────────────────────────────────────────────

_PAD = 6  # pixels of padding around each crop


def _crop(image: np.ndarray, block: Block) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    x1 = max(0, int(block.x1) - _PAD)
    y1 = max(0, int(block.y1) - _PAD)
    x2 = min(w, int(block.x2) + _PAD)
    y2 = min(h, int(block.y2) + _PAD)
    return image[y1:y2, x1:x2], (x1, y1)


def _parse_ocr_result(ocr_output: list) -> List[tuple]:
    """Normalise PaddleOCR 2.x and 3.x output → [(polygon, text, score)]."""
    lines = []
    if not ocr_output:
        return lines
    first = ocr_output[0]
    if first is None:
        return lines

    if isinstance(first, dict):   # PaddleOCR 3.x
        texts  = first.get("rec_texts",  []) or []
        scores = first.get("rec_scores", []) or []
        polys  = first.get("rec_polys",  []) or []
        for text, score, poly in zip(texts, scores, polys):
            if text and text.strip():
                lines.append((poly, text.strip(), float(score)))
        return lines

    for line in first:            # PaddleOCR 2.x
        if line is None:
            continue
        polygon, (text, score) = line
        if text and text.strip():
            lines.append((polygon, text.strip(), float(score)))
    return lines


def _poly_to_bbox(points) -> List[int]:
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    return [int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max())]


def ocr_block(image: np.ndarray, block: Block, use_gpu: bool = False) -> Optional[tuple]:
    """Run PaddleOCR on a single block crop.

    Returns:
        A tuple (text, avg_text_score) where text is the concatenated OCR output
        and avg_text_score is the mean confidence across all detected text lines.
        Returns None if no text is found or the crop is too small.
    """
    crop, _ = _crop(image, block)
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    ocr = _get_ocr_engine(use_gpu)
    try:
        result = ocr.predict(
            crop,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        result = ocr.ocr(crop, cls=False)

    lines = _parse_ocr_result(result)
    if not lines:
        return None

    lines.sort(key=lambda t: _poly_to_bbox(t[0])[1])
    text = " ".join(t[1] for t in lines).strip()
    if not text:
        return None

    # Average OCR confidence across all detected text lines in this block
    avg_text_score = round(sum(t[2] for t in lines) / len(lines), 4)
    return text, avg_text_score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Post-OCR text-level type refinement
# ─────────────────────────────────────────────────────────────────────────────

def _text_level_refine(block_type: str, text: str, block: Block) -> str:
    """
    Lightweight text-content rules applied after OCR to fix misclassifications.

    Promotes paragraph → section_header if:
      - ≤ 10 words, no sentence-ending punctuation, starts uppercase or numbered
    Promotes paragraph → title if:
      - ≤ 6 words, ALL CAPS
    Demotes section_header → paragraph if:
      - > 25 words with sentence-ending punctuation
    """
    import re

    words = text.split()
    word_count = len(words)
    ends_sentence = bool(re.search(r"[.!?]\s*$", text))
    numbered = bool(re.match(r"^(\d+\.?\d*|[A-Z]\.)[\s\u00a0]+\S", text))

    if block_type == TYPE_PARAGRAPH:
        if word_count <= 10 and not ends_sentence:
            if numbered or (words and words[0][0].isupper()):
                return TYPE_SECTION_HEADER
        if word_count <= 6 and text.strip() and text.upper() == text:
            return TYPE_TITLE

    if block_type == TYPE_SECTION_HEADER:
        if word_count > 25 and ends_sentence:
            return TYPE_PARAGRAPH

    return block_type


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PDF loading
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """Convert a PDF to a list of BGR numpy arrays, one per page."""
    print(f"  [loader] Converting PDF → images at {dpi} DPI…")
    pil_pages = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    pages = []
    for pil in pil_pages:
        arr = np.array(pil.convert("RGB"))
        pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"  [loader] {len(pages)} page(s) loaded.")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Main pipeline  (public API used by sarvam_client.py)
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    out_json: str,
    layout_filter: Optional[List[str]] = None,
    dpi: int = 300,
    use_gpu: bool = False,
    visualize: bool = False,
    vis_dir: str = "vis_output",
) -> List[Dict]:
    """
    Full pipeline: PDF → layout detection → OCR → classify → JSON.

    Args:
        pdf_path:      Path to the input PDF.
        out_json:      Path where the output JSON will be written.
        layout_filter: Optional list of canonical types to keep
                       (e.g. ["title", "section_header", "paragraph"]).
                       None → keep all text-bearing blocks.
        dpi:           PDF render resolution (300 is a good default).
        use_gpu:       Pass True to use GPU for PaddleOCR inference.
        visualize:     Write annotated debug images to vis_dir.
        vis_dir:       Directory for visualization images.

    Returns:
        List of block dicts matching the schema:
          { page, type, raw_label, score, bbox, text }
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = pdf_to_images(pdf_path, dpi=dpi)
    results: List[Dict] = []

    for page_idx, img in enumerate(pages, start=1):
        print(f"\n── Page {page_idx}/{len(pages)} ─────────────────────────────")
        page_h, page_w = img.shape[:2]

        # Step 1: Layout detection
        print("  [layout] Detecting blocks…")
        blocks = detect_blocks(img)
        print(f"  [layout] {len(blocks)} block(s) found.")

        # Step 2: Geometry-based reclassification
        _heuristic_reclassify(blocks, page_h, page_w)

        # Step 3: Reading order — top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (round(b.y1 / 20), b.x1))

        # Step 4: OCR + text-level refinement
        page_results = []
        for block in blocks:
            if layout_filter and block.type not in layout_filter:
                continue

            ocr_result = ocr_block(img, block, use_gpu=use_gpu)
            if not ocr_result:
                continue
            text, text_score = ocr_result

            final_type = _text_level_refine(block.type, text, block)

            if layout_filter and final_type not in layout_filter:
                continue

            page_results.append({
                "page":       page_idx,
                "type":       final_type,
                "raw_label":  block.raw_label,
                "bbox_score": round(block.score, 4),
                "text_score": text_score,
                "bbox":       [int(block.x1), int(block.y1),
                               int(block.x2), int(block.y2)],
                "text":       text,
            })

        results.extend(page_results)
        print(f"  [done]   {len(page_results)} block(s) extracted.")

        if visualize:
            _save_visualization(img, blocks, page_results, page_idx, vis_dir)

    # Write JSON output
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Done. {len(results)} block(s) written → {out_json}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Visualization
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    TYPE_TITLE:          (0,   140, 255),   # orange
    TYPE_SECTION_HEADER: (50,  200, 50),    # green
    TYPE_PARAGRAPH:      (200, 100, 30),    # blue-ish
    TYPE_LIST:           (180,  50, 220),   # purple
    TYPE_TABLE:          (20,  220, 220),   # cyan
    TYPE_OTHER:          (120, 120, 120),   # grey
}


def _save_visualization(
    image: np.ndarray,
    blocks: List[Block],
    page_results: List[Dict],
    page_idx: int,
    vis_dir: str,
) -> None:
    os.makedirs(vis_dir, exist_ok=True)
    vis = image.copy()

    for b in blocks:
        color = _TYPE_COLORS.get(b.type, (100, 100, 100))
        cv2.rectangle(vis, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 1)

    for entry in page_results:
        x1, y1, x2, y2 = entry["bbox"]
        btype = entry["type"]
        color = _TYPE_COLORS.get(btype, (100, 100, 100))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{btype}: {entry['text'][:40]}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out_path = os.path.join(vis_dir, f"page_{page_idx:03d}.jpg")
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  [vis]    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CLI (standalone use)
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_TYPES = [
    TYPE_TITLE, TYPE_SECTION_HEADER, TYPE_PARAGRAPH,
    TYPE_LIST, TYPE_TABLE, TYPE_OTHER,
]


def main():
    parser = argparse.ArgumentParser(
        description="Local PDF layout OCR — no API key needed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdf",       required=True,         help="Input PDF file")
    parser.add_argument("--out",       default="result.json", help="Output JSON path")
    parser.add_argument("--dpi",       default=300, type=int, help="DPI for PDF→image (default: 300)")
    parser.add_argument("--gpu",       action="store_true",   help="Use GPU for OCR")
    parser.add_argument("--visualize", action="store_true",   help="Save annotated debug images")
    parser.add_argument("--vis-dir",   default="vis_output",  help="Directory for visualization images")
    parser.add_argument(
        "--layout",
        nargs="*",
        choices=CANONICAL_TYPES,
        metavar="TYPE",
        help=(
            f"Filter output to specific block type(s). "
            f"Choices: {CANONICAL_TYPES}. "
            "Omit to extract all text-bearing blocks."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"ERROR: File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    process_pdf(
        pdf_path=args.pdf,
        out_json=args.out,
        layout_filter=args.layout or None,
        dpi=args.dpi,
        use_gpu=args.gpu,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
    )


if __name__ == "__main__":
    main()
