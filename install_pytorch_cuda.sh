#!/bin/bash
# Install PyTorch with CUDA support
# This script installs PyTorch with CUDA 12.4 support for GPU acceleration

set -e

echo "=========================================="
echo "Installing PyTorch with CUDA Support"
echo "=========================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader | head -1
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi

echo ""
echo "Installing PyTorch with CUDA 12.4 support..."
echo "This will uninstall any existing CPU-only PyTorch installation."

# Uninstall existing PyTorch (if installed)
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with CUDA 12.4
pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print("✅ CUDA support enabled!")
else:
    print("❌ CUDA not available. PyTorch is CPU-only.")
    print("   This may be due to:")
    print("   - CUDA runtime not installed")
    print("   - PyTorch CUDA version mismatch")
    print("   - Driver/runtime version mismatch")
EOF

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="

