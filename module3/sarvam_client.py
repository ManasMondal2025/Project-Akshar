"""
sarvam_client.py  (module3)
───────────────────────────
Visual-grounding metadata extractor for Module 3.

Uses the fully-local PaddleOCR pipeline (pdf_layout_ocr.py) instead of the
Sarvam Akshar API — no API key required.

Public function:
    extract_grounding_metadata(pdf_path, api_key="") → List[Dict]

The `api_key` parameter is kept for interface compatibility but is intentionally
ignored — all inference is performed locally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def extract_grounding_metadata(
    pdf_path: str,
    api_key: str = "",           # kept for backward compatibility — not used
    dpi: int = 300,
    layout_filter: List[str] | None = None,
    use_gpu: bool = False,
    visualize: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run the local layout-detection + OCR pipeline on *pdf_path* and return
    a list of block metadata dicts.

    Each dict has the schema:
        {
            "page":      int,         # 1-indexed
            "type":      str,         # title | section_header | paragraph | list | table | other
            "raw_label": str,         # original PP-DocLayout label
            "score":     float,       # detection confidence
            "bbox":      [x1,y1,x2,y2],
            "text":      str,
        }

    Args:
        pdf_path:      Path to the PDF to process.
        api_key:       Ignored — present only for interface compatibility.
        dpi:           PDF render resolution (300 DPI recommended).
        layout_filter: Restrict output to these canonical types, e.g.
                       ["title", "section_header", "paragraph"].
                       None → keep all text-bearing blocks.
        use_gpu:       Pass True to use GPU for PaddleOCR inference.
        visualize:     Write annotated debug images alongside the JSON output.

    Returns:
        List of block dicts, or an empty list on error (with a fallback
        attempt via PyMuPDF).
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Determine output JSON path (same folder as the PDF)
    base = os.path.splitext(pdf_path)[0]
    json_output_path = f"{base}_metadata.json"
    vis_dir = f"{base}_vis"

    # ── Primary path: local PaddleOCR pipeline ───────────────────────────────
    try:
        from module3.pdf_layout_ocr import process_pdf  # type: ignore

        print(f"[module3] Running local OCR pipeline on: {pdf_path}")
        results = process_pdf(
            pdf_path=pdf_path,
            out_json=json_output_path,
            layout_filter=layout_filter,
            dpi=dpi,
            use_gpu=use_gpu,
            visualize=visualize,
            vis_dir=vis_dir,
        )
        print(f"[module3] ✅ Metadata saved → {json_output_path}")
        return results

    except ImportError:
        # Try a direct relative import (when running module3 as a package)
        pass

    try:
        from pdf_layout_ocr import process_pdf  # type: ignore

        print(f"[module3] Running local OCR pipeline on: {pdf_path}")
        results = process_pdf(
            pdf_path=pdf_path,
            out_json=json_output_path,
            layout_filter=layout_filter,
            dpi=dpi,
            use_gpu=use_gpu,
            visualize=visualize,
            vis_dir=vis_dir,
        )
        print(f"[module3] ✅ Metadata saved → {json_output_path}")
        return results

    except Exception as e:
        print(f"[module3] ⚠️  Local OCR pipeline failed: {e}")

    # ── Fallback: PyMuPDF basic text extraction ───────────────────────────────
    print("[module3] Falling back to PyMuPDF basic text extraction…")
    results = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            blocks = page.get_text("blocks")
            for b in blocks:
                if b[6] == 0:  # 0 = text block
                    text_val = b[4].strip()
                    if text_val:
                        results.append({
                            "page":      page_idx + 1,
                            "type":      "paragraph",
                            "raw_label": "pymupdf_fallback",
                            "score":     1.0,
                            "bbox":      [b[0], b[1], b[2], b[3]],
                            "text":      text_val,
                        })
        doc.close()

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[module3] ✅ Fallback metadata saved → {json_output_path}")

    except Exception as fe:
        print(f"[module3] ❌ Fallback also failed: {fe}")

    return results


if __name__ == "__main__":
    pass
