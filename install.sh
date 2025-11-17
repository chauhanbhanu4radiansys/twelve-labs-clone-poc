#!/bin/bash
# Standalone installation script for Video Prism
# Run in Python 3.10 or 3.11 environment

set -e

echo "=========================================="
echo "Video Prism - Dependency Installation"
echo "=========================================="
echo ""

# Upgrade pip
echo "Step 1/4: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support FIRST (required before other packages)
echo ""
echo "Step 2/4: Installing PyTorch with CUDA support..."
echo "This may take a few minutes..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install NumPy 1.x (required for PyTorch compatibility)
echo ""
echo "Step 3/4: Installing NumPy 1.x..."
pip install --force-reinstall "numpy>=1.21.0,<2.0"

# Install all other dependencies from requirements.txt
echo ""
echo "Step 4/4: Installing remaining dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  streamlit run app_rag_chat.py"
echo ""

