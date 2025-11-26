# Keep your stable RunPod base image
FROM runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04

# Ensure we're using Python 3.12 consistently
RUN apt-get update && apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# CRITICAL: Remove conflicting CUDA compatibility libraries that cause Error 803
RUN rm -rf /usr/local/cuda*/compat/libcuda.so* || true && \
    rm -rf /usr/local/cuda*/lib64/stubs/libcuda.so* || true && \
    echo "Removed conflicting CUDA compatibility libraries"

# Install system dependencies including NVENC/NVDEC support
RUN set -e && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl tar xz-utils wget git ca-certificates \
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
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y torch torchvision torchaudio || true && \
    pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Install ImageBind from GitHub (from setup.sh)
RUN echo "Installing ImageBind from GitHub..." && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/ImageBind.git && \
    echo "✓ ImageBind installed"

# Copy requirements.txt
COPY ./requirements.txt /requirements.txt

# Install all Python dependencies from requirements.txt (from setup.sh)
RUN if [ -f /requirements.txt ]; then \
        echo "Installing Python dependencies from requirements.txt..." && \
        pip install --no-cache-dir -r /requirements.txt && \
        echo "✓ All Python dependencies installed"; \
    else \
        echo "⚠ Warning: requirements.txt not found. Skipping dependency installation."; \
    fi

# Create model cache directory
RUN mkdir -p /model_cache && \
    mkdir -p /.checkpoints

# Copy application files
COPY ./src /src
COPY ./run_example.py /handler.py

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

# Model cache directories
ENV MODEL_CACHE_DIR=/model_cache
ENV TORCH_HOME=/model_cache/torch
ENV TRANSFORMERS_CACHE=/model_cache/transformers
ENV HF_HOME=/model_cache/huggingface

CMD ["python3.12", "-u", "/handler.py"]

