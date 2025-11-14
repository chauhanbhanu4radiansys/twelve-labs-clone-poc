#!/bin/bash
# Installation script for Video Prism dependencies
# Run in Python 3.10 or 3.11 environment

set -e

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (must be installed first)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0"
pip install "pandas==2.2.2"

# Install ImageBind dependencies
pip install timm>=0.9.0 ftfy>=6.1.0 regex>=2022.0.0 einops>=0.6.0 iopath>=0.1.9

# Install ImageBind
pip install git+https://github.com/facebookresearch/ImageBind.git

# Install Hugging Face libraries (pinned for NeMo compatibility)
pip install transformers==4.35.2 huggingface_hub==0.19.4

# Install NeMo build dependencies
pip install setuptools==69.5.1 Cython>=0.29.0 pybind11
pip install --no-build-isolation youtokentome>=1.0.5

# Install NeMo core dependencies with pinned versions for stability
pip install protobuf==3.20.3 "pytorch-lightning>=2.0.0,<2.3" "hydra-core>=1.3.2,<1.4" "omegaconf>=2.3.0,<2.4"

# Install NeMo Toolkit directly from GitHub for stability (recommended)
pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]

# Install remaining dependencies
pip install \
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
