# Project Akshar

An intelligent, multi-module document processing and AI pipeline. Project Akshar handles the complete lifecycle of physical and digital documents: from image ingestion, dewarping, and layout analysis, through OCR, and finally to Vector Database indexing and RAG inference.

## 🏗 Architecture

The project is split across four distinct functional modules to separate concerns and compute requirements:

### **Module 1: Pipeline Orchestrator & Interactive Workbench**
- **Port:** `8000` (FastAPI backend)
- **Frontend:** React / Vite (runs on `5173`)
- **Responsibilities:**
  - File upload (PDF and images)
  - PDF classification (digital vs scanned) and page extraction
  - Perspective transform, page splitting, margin detection, and content bounding
  - CNN-based corner detection (`best_model_fold_5.pth`)
  - Complete pipeline orchestration calling out to Modules 3 & 4 (OCR + RAG)

### **Module 2: Advanced Image Processing & Dewarping**
- **Port:** `8001` (FastAPI backend, no separate frontend — UI served from Module 1)
- **Responsibilities:**
  - Machine Learning-based auto-dewarping (`30.pt` — xiaomore model)
  - Grid-based ScanTailor dewarping
  - Poly-dewarp curve estimation and spline correction
  - Image deskewing and adaptive/Otsu threshold enhancements

### **Module 3: Layout Detection & OCR**
- **Engine:** PaddleOCR + PaddleX (fully local, no API keys required)
- **Responsibilities:**
  - PDF layout parsing with PP-DocLayout
  - Bounding box generation and text extraction
  - PyMuPDF fallback for digital PDFs

### **Module 4: RAG & Vector Database**
- **Engine:** ChromaDB + BAAI/bge-base-en-v1.5 embeddings + Local Ollama (Llama-3)
- **Responsibilities:**
  - Generates BGE embeddings for OCR text chunks
  - Stores visual grounding metadata (bounding boxes & page numbers) in ChromaDB
  - Retrieves relevant document chunks via semantic search
  - Generates visually-grounded verifiable answers via local Ollama

---

## 🛠 Prerequisites & Installation

A `setup.sh` script is provided to automate the installation of dependencies.

### 1. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

**The setup script will:**
- Install `poppler` via Homebrew (required for `pdf2image`)
- Create and activate a Python virtual environment (`myenv1`)
- Install all Python dependencies from `requirements.txt`
- Create required local folders (`uploads/`, `processed/`, `chroma_db/`, `pretrained_models/`)
- Install Node.js dependencies for the React frontend

### 2. Download Pretrained Model Weights

Due to size constraints, ML model weights are **not** included in the repository. You must download them manually and place them in the correct paths before starting the servers.

#### **A. ML Auto-Dewarping Model (`30.pt`)**
Required by Module 2 for automatic document dewarping.
- **Source:** [xiaomore/Document-Image-Dewarping](https://github.com/xiaomore/Document-Image-Dewarping)
- **Download:** Go to the Releases section of that repo and download `30.pt`
- **Place at:**
  ```
  module2/backend/pretrained_models/30.pt
  ```

#### **B. CNN Corner Detection Model (`best_model_fold_5.pth`)**
Required by Module 1 for ML-assisted corner detection.
- **Place at:**
  ```
  module1/backend/pretrained_models/best_model_fold_5.pth
  ```

---

## 🚀 Running the Application

Open **3 terminal windows** from the project root:

### Terminal 1 — Module 1 Backend (port 8000)
```bash
source myenv1/bin/activate
cd module1/backend
uvicorn main:app --port 8000 --reload
```

### Terminal 2 — Module 2 Backend (port 8001)
```bash
source myenv1/bin/activate
cd module2/backend
uvicorn main:app --port 8001 --reload
```

### Terminal 3 — Frontend (port 5173)
```bash
cd module1/frontend
npm run dev
```

Then open your browser at: **http://localhost:5173**

---

## 📂 Project Layout

```text
.
├── README.md
├── setup.sh                          # Automated installation script
├── requirements.txt                  # Unified Python dependencies (all modules)
├── chroma_db/                        # ChromaDB vector store (auto-created)
│
├── module1/                          # Orchestrator & Image Workbench
│   └── backend/
│       ├── main.py                   # FastAPI app — port 8000
│       ├── pretrained_models/
│       │   └── best_model_fold_5.pth # ← CNN corner detection weights
│       ├── uploads/                  # Uploaded files (auto-created)
│       ├── processed/                # Shared output dir (used by M1 & M2)
│       ├── routes/
│       │   ├── upload.py
│       │   ├── corners.py
│       │   ├── transform.py
│       │   ├── scantailor.py
│       │   └── pipeline.py           # Orchestrates Modules 3 & 4
│       └── utils/
│           ├── cnn_corner_detect.py
│           ├── edge_detect_corners.py
│           ├── content_selection.py
│           ├── margins.py
│           ├── page_split.py
│           ├── page_layout.py
│           ├── transform.py
│           └── pdf_utils.py
│   └── frontend/                     # React/Vite UI — port 5173
│
├── module2/                          # ML Dewarping & Enhancement
│   └── backend/
│       ├── main.py                   # FastAPI app — port 8001
│       ├── pretrained_models/
│       │   └── 30.pt                 # ← Auto-dewarp model (xiaomore)
│       ├── routes/
│       │   ├── dewarp.py
│       │   ├── enhance.py
│       │   └── export.py             # ← Moved PDF export here
│       └── utils/
│           ├── scantailor.py
│           ├── deskew.py
│           ├── enhance.py
│           ├── dewarp_ml/            # ML dewarping (30.pt)
│           ├── poly_dewarp/          # Curve-based dewarping
│           └── pdf_utils.py          # PDF generation utilities
│
├── module3/                          # Local OCR Pipeline
│   ├── pdf_layout_ocr.py             # PaddleOCR + layout detection
│   └── sarvam_client.py             # Orchestrates OCR, PyMuPDF fallback
│
└── module4/                          # RAG & Vector Database
    ├── vector_store.py               # ChromaDB indexing (BGE embeddings)
    └── rag_inference.py              # Llama-3 via Ollama — RAG query
```
