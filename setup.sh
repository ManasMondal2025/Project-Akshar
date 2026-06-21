#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT AKSHAR — Setup & Installation Script
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "🚀 Starting setup for Project Akshar..."

# 1. Install System Dependencies (macOS)
if command -v brew >/dev/null 2>&1; then
    echo "📦 Checking system dependencies..."
    if ! brew list poppler >/dev/null 2>&1; then
        echo "Installing poppler (required for pdf2image)..."
        brew install poppler
    else
        echo "✅ Poppler is already installed."
    fi
else
    echo "⚠️ Homebrew not found. Please ensure 'poppler' is installed on your system."
fi

# 2. Setup Python Virtual Environment
VENV_NAME="myenv1"
if [ ! -d "$VENV_NAME" ]; then
    echo "🐍 Creating Python virtual environment '$VENV_NAME'..."
    python3.10 -m venv $VENV_NAME || python3 -m venv $VENV_NAME
else
    echo "✅ Virtual environment '$VENV_NAME' already exists."
fi

echo "🔄 Activating virtual environment..."
source $VENV_NAME/bin/activate

# 3. Install Python Dependencies
echo "📥 Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create required directories
echo "📂 Creating required directories..."
mkdir -p module1/backend/pretrained_models
mkdir -p module2/backend/pretrained_models
mkdir -p module1/backend/uploads
mkdir -p module1/backend/processed
mkdir -p chroma_db

echo "
⚠️  ACTION REQUIRED: Download Pretrained Weights ⚠️

To use the ML features, you must manually download the following model weights
and place them in the correct directories:

1. Auto-Dewarping Model (xiaomore)
   - Download '30.pt' from: https://github.com/xiaomore/Document-Image-Dewarping
   - Place it at: module2/backend/pretrained_models/30.pt

2. CNN Corner Detection Model
   - Place your 'best_model_fold_5.pth' model at:
     module1/backend/pretrained_models/best_model_fold_5.pth

"

# 5. Install Frontend Dependencies (Node.js)
echo "📦 Installing Node.js dependencies for the frontend..."

if [ -d "module1/frontend" ]; then
    echo "Setting up Module 1 Frontend..."
    cd module1/frontend
    npm install
    cd ../..
fi

echo "✨ Setup complete! ✨"
echo ""
echo "To run the application, open 3 terminal windows:"
echo ""
echo "  Terminal 1 (Module 1 Backend — port 8000):"
echo "    source $VENV_NAME/bin/activate && cd module1/backend && uvicorn main:app --port 8000 --reload"
echo ""
echo "  Terminal 2 (Module 2 Backend — port 8001):"
echo "    source $VENV_NAME/bin/activate && cd module2/backend && uvicorn main:app --port 8001 --reload"
echo ""
echo "  Terminal 3 (Frontend — port 5173):"
echo "    cd module1/frontend && npm run dev"
