"""
PROJECT AKSHAR - Module 2 Backend
===================================
Dedicated FastAPI service for document dewarping, enhancement, and PDF export.

Endpoints:
  - /dewarp        Grid-based ScanTailor mesh dewarp
  - /dewarp/auto   ML-based automatic dewarp (ICCV 2023 neural network)
  - /dewarp/poly   B-spline poly dewarp (dewarp2 algorithm)
  - /deskew        Auto + manual deskew
  - /enhance/*     Otsu / adaptive binarisation + colour enhancement
  - /export/pdf    Export processed images as PDF
  - /export/pdf-bbox  Export PDF with layout bounding box overlay

Runs on port 8001.
Module 1 backend (upload, pipeline, OCR, QA) runs on port 8000.

Shared processed/ folder:
  Both backends write output images to module1/backend/processed/.
  All image data is returned as base64 in JSON — no static file serving needed.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.dewarp  import router as dewarp_router
from routes.enhance import router as enhance_router
from routes.export  import router as export_router

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PROJECT AKSHAR — Module 2 API",
    description=(
        "Dewarp and enhancement service: "
        "grid dewarp, ML auto-dewarp, poly dewarp, deskew, and image enhancement."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — Allow the React dev server (Vite) on common ports
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(dewarp_router,  tags=["Dewarp & Deskew"])
app.include_router(enhance_router, tags=["Enhancement"])
app.include_router(export_router,  tags=["Export"])

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "service": "module2",
        "version": "1.0.0",
        "status": "running",
        "port": 8001,
        "endpoints": {
            "dewarp": {
                "grid":       "POST /dewarp",
                "auto_ml":    "POST /dewarp/auto",
                "poly":       "POST /dewarp/poly",
                "analyze":    "POST /dewarp/analyze-grid",
                "poly_est":   "POST /dewarp/poly/estimate-curves",
                "deskew":     "POST /deskew",
                "deskew_man": "POST /deskew/manual",
            },
            "enhance": {
                "otsu":     "POST /enhance/otsu",
                "adaptive": "POST /enhance/adaptive",
            },
            "export": {
                "pdf":      "POST /export/pdf",
                "pdf_bbox": "POST /export/pdf-bbox",
            },
        },
    }

# ---------------------------------------------------------------------------
# Dev Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
