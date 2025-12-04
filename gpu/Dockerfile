# Keep your stable RunPod base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Use Python 3.11 from base image (already installed)
RUN python3.11 -m pip install --upgrade pip

# CRITICAL: Remove conflicting CUDA compatibility libraries that cause Error 803
RUN rm -rf /usr/local/cuda*/compat/libcuda.so* || true && \
    rm -rf /usr/local/cuda*/lib64/stubs/libcuda.so* || true && \
    echo "Removed conflicting CUDA compatibility libraries"

# Install system dependencies including NVENC/NVDEC support
RUN set -e && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl tar xz-utils wget git ca-certificates \
    dnsutils iputils-ping \
    # CUDA runtime libraries
    libcudnn8 \
    libcublas-12-4 \
    libcufft-12-4 \
    libcurand-12-4 \
    libcusolver-12-4 \
    libcusparse-12-4 \
    libnpp-12-4 \
    # Enhanced FFmpeg with NVENC/NVDEC support
    ffmpeg \
    # Video/image processing libraries
    libjpeg8 libpng16-16 libtiff5 libopenexr25 libwebp7 \
    libgtk-3-0 libtbb12 \
    libv4l-0 libxvidcore4 libx264-163 libx265-199 \
    libavcodec58 libavformat58 libswscale5 libswresample3 \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libprotobuf23 \
    # GPU acceleration libraries for video
    libva-dev libvdpau-dev \
    # Essential runtime
    libgl1-mesa-glx libxext6 libsm6 libxrender1 libglib2.0-0 \
    # Python essentials
    python3-numpy python3-dev \
    # Development tools for compilation
    pkg-config \
    libavutil-dev libavcodec-dev libavformat-dev \
    libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CUDA environment setup with video capabilities
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install PyTorch 2.4.0 with CUDA 12.4 (from setup.sh)
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip uninstall -y torch torchvision torchaudio || true && \
    python3.11 -m pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Install ImageBind from GitHub (from setup.sh)
RUN echo "Installing ImageBind from GitHub..." && \
    python3.11 -m pip install --no-cache-dir git+https://github.com/facebookresearch/ImageBind.git && \
    echo "✓ ImageBind installed"

# Copy requirements.txt
COPY ./gpu/requirements.txt /requirements.txt

# Install all Python dependencies from requirements.txt (from setup.sh)
RUN if [ -f /requirements.txt ]; then \
        echo "Installing Python dependencies from requirements.txt..." && \
        python3.11 -m pip install --no-cache-dir -r /requirements.txt && \
        echo "✓ All Python dependencies installed"; \
    else \
        echo "⚠ Warning: requirements.txt not found. Skipping dependency installation."; \
    fi

# Create model cache directories
RUN mkdir -p /model_cache && \
    mkdir -p /.checkpoints && \
    mkdir -p /model_cache/torch && \
    mkdir -p /model_cache/transformers && \
    mkdir -p /model_cache/huggingface

# Set model cache environment variables BEFORE downloading models
ENV MODEL_CACHE_DIR=/model_cache
ENV TORCH_HOME=/model_cache/torch
ENV TRANSFORMERS_CACHE=/model_cache/transformers
ENV HF_HOME=/model_cache/huggingface

# Pre-download models during build time to avoid cold start delays
# This ensures models are available immediately when Lambda starts
RUN echo "==========================================" && \
    echo "Pre-downloading models for faster cold starts" && \
    echo "==========================================" && \
    echo ""

# Pre-download ImageBind model (~5GB) to /.checkpoints/imagebind_huge.pth
RUN echo "Pre-downloading ImageBind model (this may take several minutes)..." && \
    python3.11 -c "import os; \
    import sys; \
    sys.path.insert(0, '/'); \
    from imagebind.models import imagebind_model; \
    import torch; \
    print('Downloading ImageBind model...'); \
    model = imagebind_model.imagebind_huge(pretrained=True); \
    print('Saving ImageBind checkpoint...'); \
    os.makedirs('/.checkpoints', exist_ok=True); \
    torch.save(model.state_dict(), '/.checkpoints/imagebind_huge.pth'); \
    print('✓ ImageBind model downloaded and saved to /.checkpoints/imagebind_huge.pth'); \
    print(f'Checkpoint size: {os.path.getsize(\"/.checkpoints/imagebind_huge.pth\") / (1024**3):.2f} GB')" && \
    echo "✓ ImageBind model pre-downloaded successfully"

# Pre-download BLIP captioning model (~2GB) to HuggingFace cache
RUN echo "Pre-downloading BLIP captioning model (this may take several minutes)..." && \
    python3.11 -c "import os; \
    os.environ['HF_HOME'] = '/model_cache/huggingface'; \
    os.environ['TRANSFORMERS_CACHE'] = '/model_cache/transformers'; \
    from transformers import BlipProcessor, BlipForConditionalGeneration; \
    print('Downloading BLIP processor...'); \
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir='/model_cache/huggingface'); \
    print('Downloading BLIP model...'); \
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir='/model_cache/huggingface'); \
    print('✓ BLIP model downloaded and cached'); \
    cache_size = sum(os.path.getsize(os.path.join(dirpath, filename)) \
                     for dirpath, dirnames, filenames in os.walk('/model_cache/huggingface') \
                     for filename in filenames) / (1024**3); \
    print(f'BLIP cache size: {cache_size:.2f} GB')" && \
    echo "✓ BLIP model pre-downloaded successfully" && \
    echo "" && \
    echo "==========================================" && \
    echo "All models pre-downloaded successfully!" && \
    echo "=========================================="

# Copy application files
COPY ../src /src
COPY ./gpu/handler.py /handler.py
COPY ./gpu/local.py /local.py

# Performance environment variables
ENV PYTHONPATH=/
ENV QT_QPA_PLATFORM=offscreen
ENV CUDA_CACHE_DISABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Video processing optimization environment variables
ENV VFXAI_ENABLE_NVDEC=1
ENV VFXAI_ENABLE_NVENC=1
ENV VFXAI_FFMPEG_DECODE_FLAGS="-hwaccel cuda -hwaccel_output_format cuda"
ENV VFXAI_FFMPEG_ENCODE_FLAGS="-c:v h264_nvenc -preset p4 -cq 25 -pix_fmt yuv420p"
ENV VFXAI_NVENC_FALLBACK_ENABLED=1

CMD ["python3.11", "-u", "/handler.py"]

