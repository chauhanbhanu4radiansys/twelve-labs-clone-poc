#!/bin/bash

# Setup script for embedding pipeline
# This script installs all dependencies including ImageBind from GitHub
# Note: ffmpeg is assumed to be already installed

# Continue on error for verification step

echo "=========================================="
echo "Embedding Pipeline Setup"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ Error: pip not found. Please install pip first."
    exit 1
fi

PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "✓ Using pip: $PIP_CMD"
echo ""

# Check if ffmpeg is installed (informational only)
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg is installed: $(ffmpeg -version | head -n1)"
else
    echo "⚠ Warning: ffmpeg not found. Please install ffmpeg system-wide."
    echo "  macOS: brew install ffmpeg"
    echo "  Linux: sudo apt-get install ffmpeg"
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install PyTorch first (required for ImageBind)
echo "Installing PyTorch..."
$PIP_CMD install torch torchvision torchaudio --quiet
echo "✓ PyTorch installed"
echo ""

# Install ImageBind from GitHub
echo "Installing ImageBind from GitHub..."
$PIP_CMD install git+https://github.com/facebookresearch/ImageBind.git
echo "✓ ImageBind installed"
echo ""

# Install all other Python dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    $PIP_CMD install -r requirements.txt
    echo "✓ All Python dependencies installed"
else
    echo "⚠ Warning: requirements.txt not found. Skipping dependency installation."
fi
echo ""

# Verify critical packages
echo "Verifying installation..."
python3 -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ❌ PyTorch not found"
python3 -c "import imagebind; print('  ✓ ImageBind installed')" 2>/dev/null || echo "  ❌ ImageBind not found"
python3 -c "import transformers; print(f'  ✓ Transformers: {transformers.__version__}')" 2>/dev/null || echo "  ❌ Transformers not found"
python3 -c "import cv2; print(f'  ✓ OpenCV: {cv2.__version__}')" 2>/dev/null || echo "  ❌ OpenCV not found"
python3 -c "import pinecone; print('  ✓ Pinecone installed')" 2>/dev/null || echo "  ❌ Pinecone not found"
python3 -c "import requests; print(f'  ✓ Requests: {requests.__version__}')" 2>/dev/null || echo "  ❌ Requests not found"
python3 -c "from scenedetect import SceneManager; print('  ✓ PySceneDetect installed')" 2>/dev/null || echo "  ❌ PySceneDetect not found"
python3 -c "from pydub import AudioSegment; print('  ✓ Pydub installed')" 2>/dev/null || echo "  ❌ Pydub not found"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update Pinecone credentials in src/embedding/config.py"
echo "2. Create Pinecone indexes (video-search, audio-search, text-search, desc-search)"
echo "3. Run the pipeline: python src/embedding/run_example.py"
echo ""

