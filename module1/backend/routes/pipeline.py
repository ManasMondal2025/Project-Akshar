"""
PROJECT AKSHAR - Pipeline Orchestration Router
==================================================
Manages the full AI document processing pipeline:
  1. Upload PDF or image
  2. Detect whether PDF is digital or scanned
  3. Extract pages from scanned PDFs
  4. Run Module 3 (OCR / text layout extraction)
  5. Run Module 4 (ChromaDB indexing)
  6. RAG Query (answer + references)
"""

import os
import sys
import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root so module3/module4 are importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
WORKSPACE_ROOT = os.path.join(os.path.dirname(BASE_DIR), "..")

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../../")))

from utils.pdf_utils import detect_pdf_type, extract_pdf_pages, images_to_pdf

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    pdf_path: str


class ExtractPagesRequest(BaseModel):
    pdf_path: str


class ConvertToPdfRequest(BaseModel):
    image_paths: List[str]
    output_filename: str = "assembled.pdf"


class RunOcrRequest(BaseModel):
    pdf_path: str
    sarvam_api_key: str = ""


class IndexRequest(BaseModel):
    document_id: str
    metadata_blocks: List[dict]
    collection_name: str = "project_akshar"


class QueryRequest(BaseModel):
    query: str
    document_id: str
    collection_name: str = "project_akshar"
    groq_api_key: str = ""


class RegisterPageRequest(BaseModel):
    """Register an already-extracted server-side image for the workbench."""
    server_path: str
    label: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def pipeline_upload(file: UploadFile = File(...)):
    """
    Accept a PDF or image upload and save it to the uploads directory.
    Returns file path and detected file type (pdf/image).
    """
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename or "file")[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    file_type = "pdf" if ext == ".pdf" else "image"

    return {
        "status": "success",
        "file_path": save_path,
        "file_type": file_type,
        "original_filename": file.filename,
        "file_id": file_id,
    }


@router.post("/detect")
async def detect_document_type(request: DetectRequest):
    """
    Detect whether a PDF is digital (text-based) or scanned (image-based).
    Uses PyMuPDF text density analysis.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        doc_type = detect_pdf_type(request.pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # Determine pipeline routing
    route = "direct_ocr" if doc_type == "digital" else "image_workbench"

    return {
        "status": "success",
        "pdf_path": request.pdf_path,
        "document_type": doc_type,
        "recommended_route": route,
        "message": (
            "Digital PDF detected — routing directly to QA pipeline (Module 3 → 4)."
            if doc_type == "digital"
            else "Scanned PDF detected — routing through Image Workbench (Module 1 → 3 → 4)."
        )
    }


@router.post("/extract-pages")
async def extract_pdf_pages_endpoint(request: ExtractPagesRequest):
    """
    Extract each page of a scanned PDF as a PNG image.
    Returns ordered list of image paths for feeding into Module 1.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    output_dir = os.path.join(PROCESSED_DIR, f"pages_{uuid.uuid4().hex[:8]}")

    try:
        image_paths = extract_pdf_pages(request.pdf_path, output_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Page extraction failed: {str(e)}")

    return {
        "status": "success",
        "page_count": len(image_paths),
        "image_paths": image_paths,
    }


@router.post("/convert-to-pdf")
async def convert_images_to_pdf(request: ConvertToPdfRequest):
    """
    Assemble ordered processed images into a single PDF.
    Call this after Module 1 processing to produce the clean output PDF.
    """
    for path in request.image_paths:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    output_path = os.path.join(PROCESSED_DIR, f"{uuid.uuid4().hex}_{request.output_filename}")

    try:
        images_to_pdf(request.image_paths, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF assembly failed: {str(e)}")

    return {
        "status": "success",
        "pdf_path": output_path,
        "page_count": len(request.image_paths),
    }


@router.post("/run-ocr")
async def run_ocr(request: RunOcrRequest):
    """
    Run Module 3 OCR on a PDF file.
    Tries Sarvam Akshar API first; falls back to PyMuPDF local extraction.
    Returns structured text blocks with bounding boxes and page numbers.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        # Add module3 to path
        module3_path = os.path.abspath(os.path.join(BASE_DIR, "../../module3"))
        if module3_path not in sys.path:
            sys.path.insert(0, os.path.dirname(module3_path))

        from module3.sarvam_client import extract_grounding_metadata

        api_key = request.sarvam_api_key or "sk_p8ojpatk_S4tt4EfsJQJPgW6evwiWkgMY"
        blocks = extract_grounding_metadata(request.pdf_path, api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    return {
        "status": "success",
        "pdf_path": request.pdf_path,
        "block_count": len(blocks),
        "blocks": blocks,
    }


@router.post("/run-ocr-sarvam")
async def run_ocr_sarvam(request: RunOcrRequest):
    """
    Run Document Parse via the actual Sarvam AI API using the sarvamai python SDK.
    Returns structured text blocks compatible with Module 4 vector store.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    import zipfile
    import json
    import tempfile
    import glob

    api_key = request.sarvam_api_key or "sk_p8ojpatk_S4tt4EfsJQJPgW6evwiWkgMY"
    
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=api_key)
        
        job = client.document_intelligence.create_job(
            language="hi-IN",
            output_format="md"
        )
        
        job.upload_file(request.pdf_path)
        job.start()
        status = job.wait_until_complete()
        
        if status.job_state.lower() not in ["completed", "success"]:
            raise Exception(f"Job failed with status: {status.job_state}")

        blocks = []
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "output.zip")
            job.download_output(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Find all JSON metadata files
            json_files = glob.glob(os.path.join(tmpdir, "**", "*.json"), recursive=True)
            for jf in sorted(json_files):
                with open(jf, "r") as f:
                    page_data = json.load(f)
                    page_num = page_data.get("page_num", 1)
                    
                    for block in page_data.get("blocks", []):
                        coords = block.get("coordinates", {})
                        if not coords:
                            continue
                        
                        blocks.append({
                            "page": page_num,
                            "type": block.get("layout_tag", "paragraph"),
                            "score": block.get("confidence", 1.0),
                            "bbox": [
                                coords.get("x1", 0),
                                coords.get("y1", 0),
                                coords.get("x2", 0),
                                coords.get("y2", 0)
                            ],
                            "text": block.get("text", "")
                        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sarvam API failed: {str(e)}")

    return {
        "status": "success",
        "pdf_path": request.pdf_path,
        "block_count": len(blocks),
        "blocks": blocks,
    }


@router.post("/index")
async def index_document(request: IndexRequest):
    """
    Index OCR-extracted text blocks into ChromaDB for RAG retrieval.
    """
    try:
        module4_path = os.path.abspath(os.path.join(BASE_DIR, "../../"))
        if module4_path not in sys.path:
            sys.path.insert(0, module4_path)

        from module4.vector_store import build_vector_index

        chroma_dir = os.path.abspath(os.path.join(BASE_DIR, "../../chroma_db"))
        collection = build_vector_index(
            request.document_id,
            request.metadata_blocks,
            request.collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return {
        "status": "success",
        "document_id": request.document_id,
        "indexed_blocks": len(request.metadata_blocks),
        "collection": request.collection_name,
    }


@router.post("/query")
async def rag_query(request: QueryRequest):
    """
    Run a RAG query against indexed document content.
    Returns:
      - answer: Generated answer from Llama-3 via Groq
      - references: List of {text, page_num} — text snippets only, NO bbox coords
    """
    try:
        module4_path = os.path.abspath(os.path.join(BASE_DIR, "../../"))
        if module4_path not in sys.path:
            sys.path.insert(0, module4_path)

        from module4.rag_inference import answer_query

        api_key = request.groq_api_key or "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"
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY not provided. Set via request body or environment variable."
            )

        result = answer_query(
            query=request.query,
            document_id=request.document_id,
            collection_name=request.collection_name,
            api_key=api_key,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Build references — only text + page_num + bbox
    references = []
    for h in result.get("highlights", []):
        references.append({
            "text": h.get("text", ""),
            "page_num": h.get("page_num", 1),
            "bbox": h.get("bbox", []),
        })

    return {
        "status": "success",
        "query": request.query,
        "answer": result.get("answer", ""),
        "references": references,
    }


@router.post("/register-page")
async def register_page(request: RegisterPageRequest):
    """
    Register an already-extracted server-side image file into the workbench.
    Reads the file, generates a base64 preview, and returns the standard
    { file_path, preview, width, height } shape — identical to /upload.

    Used when scanned PDF pages have already been extracted server-side and
    the frontend needs to load them without re-sending file bytes.
    """
    import base64
    import cv2

    path = request.server_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {path}")

    h, w = img.shape[:2]
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status":    "success",
        "file_path": path,
        "file_type": "image",
        "preview":   f"data:image/png;base64,{b64}",
        "width":     w,
        "height":    h,
    }
