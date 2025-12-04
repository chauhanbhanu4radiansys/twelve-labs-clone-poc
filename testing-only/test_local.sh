#!/bin/bash

# Test script for running local.py with local video and transcript files via Docker
# Usage: ./test_local.sh [video_file] [transcript_file] [video_name]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="video-search-gpu"
IMAGE_TAG="latest"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values or use command line arguments
VIDEO_FILE="${1:-}"
TRANSCRIPT_FILE="${2:-}"
VIDEO_NAME="${3:-test_video}"

# If not provided, try to find common file patterns
if [ -z "$VIDEO_FILE" ]; then
    echo -e "${YELLOW}No video file specified. Searching for video files...${NC}"
    VIDEO_FILE=$(find . -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | head -1)
fi

if [ -z "$TRANSCRIPT_FILE" ]; then
    echo -e "${YELLOW}No transcript file specified. Searching for transcript files...${NC}"
    TRANSCRIPT_FILE=$(find . -maxdepth 1 -type f \( -name "*.json" -o -name "*transcript*.json" -o -name "*transcript*.txt" \) | head -1)
fi

# Validate files exist
if [ -z "$VIDEO_FILE" ] || [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${RED}Error: Video file not found!${NC}"
    echo ""
    echo "Usage:"
    echo "  ./test_local.sh <video_file> <transcript_file> [video_name]"
    echo ""
    echo "Or place video.mp4 and transcript.json in the video-search directory"
    echo "and run: ./test_local.sh"
    exit 1
fi

if [ -z "$TRANSCRIPT_FILE" ] || [ ! -f "$TRANSCRIPT_FILE" ]; then
    echo -e "${RED}Error: Transcript file not found!${NC}"
    echo ""
    echo "Usage:"
    echo "  ./test_local.sh <video_file> <transcript_file> [video_name]"
    echo ""
    echo "Or place video.mp4 and transcript.json in the video-search directory"
    echo "and run: ./test_local.sh"
    exit 1
fi

# Convert to absolute paths
VIDEO_PATH=$(realpath "$VIDEO_FILE")
TRANSCRIPT_PATH=$(realpath "$TRANSCRIPT_FILE")
ENV_PATH=$(realpath ".env" 2>/dev/null || echo "")

# Check if Docker image exists, if not, build it
if ! docker images | grep -q "^${IMAGE_NAME}.*${IMAGE_TAG}"; then
    echo -e "${YELLOW}Docker image ${IMAGE_NAME}:${IMAGE_TAG} not found.${NC}"
    echo -e "${YELLOW}Building Docker image...${NC}"
    echo ""
    ./test_docker_build.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build Docker image. Exiting.${NC}"
        exit 1
    fi
    echo ""
else
    # Check if local.py exists in image, if not, we'll mount it
    if ! docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" test -f /local.py 2>/dev/null; then
        if [ -z "$LOCAL_PY_PATH" ] || [ ! -f "$LOCAL_PY_PATH" ]; then
            echo -e "${YELLOW}Warning: /local.py not found in image and local file not found.${NC}"
            echo -e "${YELLOW}Rebuilding image to include local.py...${NC}"
            echo ""
            ./test_docker_build.sh
            if [ $? -ne 0 ]; then
                echo -e "${RED}Failed to build Docker image. Exiting.${NC}"
                exit 1
            fi
            echo ""
        else
            echo -e "${GREEN}Mounting local.py from host (no rebuild needed)${NC}"
        fi
    fi
fi

# Check if .env file exists
if [ -z "$ENV_PATH" ] || [ ! -f "$ENV_PATH" ]; then
    echo -e "${YELLOW}Warning: .env file not found in ${SCRIPT_DIR}${NC}"
    echo -e "${YELLOW}Environment variables will need to be set in the container${NC}"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Testing local.py with Docker${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Docker Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Video file: $VIDEO_PATH"
echo "Transcript file: $TRANSCRIPT_PATH"
echo "Video name: $VIDEO_NAME"
if [ -n "$ENV_PATH" ]; then
    echo ".env file: $ENV_PATH"
fi
echo ""

# Get local.py and handler.py paths
LOCAL_PY_PATH=$(realpath "gpu/local.py" 2>/dev/null || echo "")
HANDLER_PY_PATH=$(realpath "gpu/handler.py" 2>/dev/null || echo "")

# Prepare Docker volume mounts
VOLUME_MOUNTS=(
    "-v" "$VIDEO_PATH:/mnt/video:ro"
    "-v" "$TRANSCRIPT_PATH:/mnt/transcript:ro"
)

# Mount local.py if it exists (for development/testing without rebuilding)
if [ -n "$LOCAL_PY_PATH" ] && [ -f "$LOCAL_PY_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$LOCAL_PY_PATH:/local.py:ro")
fi

# Mount handler.py if it exists (to use latest version without rebuilding)
if [ -n "$HANDLER_PY_PATH" ] && [ -f "$HANDLER_PY_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$HANDLER_PY_PATH:/handler.py:ro")
fi

# Mount .env file if it exists
if [ -n "$ENV_PATH" ] && [ -f "$ENV_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$ENV_PATH:/.env:ro")
fi

# Check for GPU availability and Docker GPU support
GPU_FLAG=""
HAS_GPU=false
HAS_DOCKER_GPU=false

# Check if nvidia-smi is available (GPU hardware exists)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    HAS_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ GPU hardware detected: ${GPU_NAME}${NC}"
    
    # Test if Docker can access GPU (suppress all output)
    echo -e "${YELLOW}Testing Docker GPU access...${NC}"
    if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        HAS_DOCKER_GPU=true
        echo -e "${GREEN}✓ Docker GPU support available - using GPU acceleration${NC}"
    else
        echo -e "${RED}✗ CRITICAL: Docker GPU support NOT available!${NC}"
        echo ""
        echo -e "${RED}Video processing requires GPU and will be extremely slow on CPU.${NC}"
        echo -e "${RED}Please install NVIDIA Container Toolkit before proceeding.${NC}"
        echo ""
        echo -e "${YELLOW}Installation instructions (Ubuntu/Debian):${NC}"
        echo ""
        echo "distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\"
        echo "  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
        echo "  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "sudo systemctl restart docker"
        echo ""
        echo -e "${RED}After installation, run this script again.${NC}"
        echo ""
        echo -e "${YELLOW}To test GPU access manually:${NC}"
        echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
        echo ""
        exit 1
    fi
else
    echo -e "${RED}✗ CRITICAL: No GPU hardware detected!${NC}"
    echo ""
    echo -e "${RED}Video processing requires GPU and will be extremely slow on CPU.${NC}"
    echo -e "${RED}Please ensure:${NC}"
    echo "  1. NVIDIA GPU is installed"
    echo "  2. NVIDIA drivers are installed (run: nvidia-smi)"
    echo "  3. NVIDIA Container Toolkit is installed"
    echo ""
    exit 1
fi
echo ""

# Run local.py inside Docker container
echo -e "${GREEN}Starting Docker container...${NC}"
echo ""

# Build docker command with conditional GPU flag
DOCKER_CMD=("docker" "run" "--rm" "-it")

# Only add GPU flag if Docker GPU support is available
if [ "$HAS_DOCKER_GPU" = true ]; then
    DOCKER_CMD+=("--gpus" "all")
fi

# Add volume mounts
DOCKER_CMD+=("${VOLUME_MOUNTS[@]}")

# Add environment variable
DOCKER_CMD+=("-e" "RUNTIME_ENVIRONMENT=local")

# Add image and command
DOCKER_CMD+=("${IMAGE_NAME}:${IMAGE_TAG}")
DOCKER_CMD+=("python3.11" "/local.py" "/mnt/video" "/mnt/transcript" "$VIDEO_NAME")

# Execute the command
"${DOCKER_CMD[@]}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Container execution completed${NC}"
echo -e "${GREEN}========================================${NC}"
