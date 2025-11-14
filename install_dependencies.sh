#!/bin/bash
# ============================================================================
# Single Installation Script for Video Prism
# ============================================================================
# This script installs all dependencies in the correct order to prevent
# build errors related to torch, Cython, and version conflicts.
# It should be run in an environment with Python 3.10 or 3.11.
# ============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.

echo "▶ Step 1/8: Upgrading pip, setuptools, and wheel..."
pip install --upgrade --verbose pip setuptools wheel

echo "▶ Step 2/8: Installing Core ML Libraries (PyTorch)..."
# PyTorch is a build dependency for several other packages and must be installed first.
# This is a large download (several GB) and may take 10-30 minutes depending on your connection.
# Using conda if available is faster, otherwise fall back to pip.
if command -v conda &> /dev/null; then
    echo "   Using conda for PyTorch installation (faster and more reliable)..."
    conda install -v pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y || \
    conda install -v pytorch torchvision torchaudio cpuonly -c pytorch -y
    pip install --verbose "numpy<2.0"
else
    echo "   Using pip for PyTorch installation (this may take a while)..."
    pip install --verbose --progress-bar pretty torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 "numpy<2.0"
fi
echo "   ✓ PyTorch installation complete"

echo "▶ Step 3/8: Installing ImageBind dependencies..."
pip install --verbose --progress-bar pretty timm>=0.9.0 ftfy>=6.1.0 regex>=2022.0.0 einops>=0.6.0 iopath>=0.1.9

echo "▶ Step 4/8: Installing ImageBind from GitHub..."
# This must be done after torch is installed.
pip install --verbose --progress-bar pretty git+https://github.com/facebookresearch/ImageBind.git

echo "▶ Step 5/8: Installing pinned Hugging Face libraries for NeMo compatibility..."
# NeMo requires huggingface_hub < 0.20.0 for 'ModelFilter'.
# transformers==4.35.2 is compatible with that version of huggingface_hub.
pip install --verbose --progress-bar pretty transformers==4.35.2 huggingface_hub==0.19.4

echo "▶ Step 6/8: Installing NeMo build dependencies..."
# Cython is needed to build youtokentome, and both are needed for NeMo.
pip install --verbose --progress-bar pretty Cython>=0.29.0 youtokentome>=1.0.5

echo "▶ Step 7/8: Installing NeMo Toolkit for ASR..."
# We use [asr] to avoid broken dependencies in [all] (e.g., megatron-core).
# --no-build-isolation is crucial to ensure it uses the already-installed Cython.
pip install --verbose --progress-bar pretty --no-build-isolation "nemo_toolkit[asr]>=1.20.0"

echo "▶ Step 8/8: Installing remaining application dependencies..."
pip install --verbose --progress-bar pretty \
    opencv-python>=4.8.0 \
    pillow>=9.0.0 \
    "scenedetect[opencv]>=0.6.2" \
    openai-whisper>=20230918 \
    pydub>=0.25.1 \
    pinecone>=3.0.0 \
    streamlit>=1.28.0 \
    boto3>=1.28.0 \
    pymongo>=4.5.0 \
    openai>=1.0.0 \
    typing-extensions>=4.5.0

echo "-----------------------------------"
echo "✅ Installation complete!"
echo "-----------------------------------"
