"""
PROJECT AKSHAR - Module 1 Backend
====================================
Image workbench, pipeline orchestration, OCR and QA.

  Port 8000  — this service
  Port 8001  — Module 2 (dewarp, deskew, enhancement)

Endpoints served here:
  - Upload / transform / corners / split / margins / content / export
  - Pipeline orchestration (detect, extract pages, assemble PDF)
  - Module 3: OCR Layout Extraction
  - Module 4: ChromaDB RAG QA
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.upload        import router as upload_router
from routes.transform     import router as transform_router
from routes.pipeline      import router as pipeline_router
from routes.scantailor    import router as scantailor_router
from routes.corners       import router as corners_router

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PROJECT AKSHAR API",
    description=(
        "Full-stack AI document processing system: "
        "image workbench, OCR layout extraction, and RAG-based QA with references."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — Allow React dev server on common ports
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static File Directories
# ---------------------------------------------------------------------------

BACKEND_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
PROCESSED_DIR = os.path.join(BACKEND_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health check and endpoint directory."""
    return {
        "project": "PROJECT AKSHAR",
        "version": "2.0.0",
        "status": "running",
        "note": "Dewarp + enhance have moved to Module 2 — port 8001",
        "modules": {
            "module1_image_workbench": {
                "upload":          "POST /upload",
                "transform":       "POST /transform",
                "corners_detect":  "POST /corners/detect",
                "corners_apply":   "POST /corners/apply",
            },
            "module2_dewarp_enhance": {
                "note":            "Served by module2 backend on port 8001",
                "dewarp":          "POST :8001/dewarp",
                "deskew":          "POST :8001/deskew",
                "enhance":         "POST :8001/enhance/otsu",
            },
            "pipeline_orchestration": {
                "upload":          "POST /pipeline/upload",
                "detect_type":     "POST /pipeline/detect",
                "extract_pages":   "POST /pipeline/extract-pages",
                "convert_to_pdf":  "POST /pipeline/convert-to-pdf",
                "run_ocr":         "POST /pipeline/run-ocr",
                "index":           "POST /pipeline/index",
                "query":           "POST /pipeline/query",
            },
        },
        "docs": "/docs",
    }

# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------

app.include_router(upload_router,    tags=["Upload"])
app.include_router(transform_router, tags=["Transform"])
app.include_router(pipeline_router)
app.include_router(scantailor_router,  tags=["ScanTailor"])
app.include_router(corners_router,     tags=["Corner Detection"])

# ---------------------------------------------------------------------------
# Dev Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
