#!/bin/bash

# Test script for running local.py with local video and transcript files via Docker
# Supports embedding, search, and analyse functionality
#
# Usage for EMBEDDING:
#   ./test_local.sh [video_file] [transcript_file] [video_name]
#   ./test_local.sh embed [video_file] [transcript_file] [video_name]
#
# Usage for SEARCH:
#   ./test_local.sh search text "your search query" [top_k]
#   ./test_local.sh search image "https://example.com/image.jpg" [top_k]
#   ./test_local.sh search audio "https://example.com/audio.wav" [top_k]
#
# Usage for ANALYSE:
#   ./test_local.sh analyse <attachment_id> "your analysis query"
#   ./test_local.sh analyze <attachment_id> "your analysis query"
#
# Options:
#   --no-build    Skip Docker image build (use existing image or fail if not found)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="video-search-gpu"
IMAGE_TAG="latest"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for --no-build flag
SKIP_BUILD=false
if [ "$1" = "--no-build" ]; then
    SKIP_BUILD=true
    shift  # Remove --no-build flag
fi

# Determine task type from first argument
TASK_TYPE="embedding"
if [ "$1" = "search" ] || [ "$1" = "embed" ] || [ "$1" = "embedding" ] || [ "$1" = "analyse" ] || [ "$1" = "analyze" ]; then
    TASK_TYPE="$1"
    shift  # Remove task type from arguments
fi

# Normalize analyse/analyze to "analyse"
if [ "$TASK_TYPE" = "analyze" ]; then
    TASK_TYPE="analyse"
fi

# If first arg is not a recognized task type, assume it's a video file (embedding mode)
if [ "$TASK_TYPE" != "search" ] && [ "$TASK_TYPE" != "embed" ] && [ "$TASK_TYPE" != "embedding" ] && [ "$TASK_TYPE" != "analyse" ]; then
    TASK_TYPE="embedding"
fi

# Parse arguments based on task type
if [ "$TASK_TYPE" = "search" ]; then
    # Search mode: search <search_type> <query> [top_k]
    SEARCH_TYPE="${1:-}"
    QUERY="${2:-}"
    TOP_K="${3:-8}"
    
    if [ -z "$SEARCH_TYPE" ] || [ -z "$QUERY" ]; then
        echo -e "${RED}Error: Search mode requires search_type and query${NC}"
        echo ""
        echo "Usage for SEARCH:"
        echo "  ./test_local.sh search text \"your search query\" [top_k]"
        echo "  ./test_local.sh search image \"https://example.com/image.jpg\" [top_k]"
        echo "  ./test_local.sh search audio \"https://example.com/audio.wav\" [top_k]"
        echo ""
        echo "Examples:"
        echo "  ./test_local.sh search text \"a person running\""
        echo "  ./test_local.sh search image \"https://example.com/image.jpg\" 10"
        echo "  ./test_local.sh search audio \"https://example.com/audio.wav\""
        exit 1
    fi
    
    if [ "$SEARCH_TYPE" != "text" ] && [ "$SEARCH_TYPE" != "image" ] && [ "$SEARCH_TYPE" != "audio" ]; then
        echo -e "${RED}Error: Invalid search_type: $SEARCH_TYPE${NC}"
        echo "Valid search types: text, image, audio"
        exit 1
    fi
elif [ "$TASK_TYPE" = "analyse" ]; then
    # Analyse mode: analyse <attachment_id> <query>
    ATTACHMENT_ID="${1:-}"
    QUERY="${2:-}"
    
    if [ -z "$ATTACHMENT_ID" ] || [ -z "$QUERY" ]; then
        echo -e "${RED}Error: Analyse mode requires attachment_id and query${NC}"
        echo ""
        echo "Usage for ANALYSE:"
        echo "  ./test_local.sh analyse <attachment_id> \"your analysis query\""
        echo "  ./test_local.sh analyze <attachment_id> \"your analysis query\""
        echo ""
        echo "Examples:"
        echo "  ./test_local.sh analyse 69326bd6f206715c442b4c1c \"What are the key moments in this video?\""
        echo "  ./test_local.sh analyse 69326bd6f206715c442b4c1c \"Summarize this video\""
        echo "  ./test_local.sh analyse 69326bd6f206715c442b4c1c \"Generate hashtags and topics\""
        exit 1
    fi
else
    # Embedding mode: [video_file] [transcript_file] [video_name]
    VIDEO_FILE="${1:-}"
    TRANSCRIPT_FILE="${2:-}"
    VIDEO_NAME="${3:-test_video}"
fi

# Handle embedding mode file validation
if [ "$TASK_TYPE" = "embedding" ] || [ "$TASK_TYPE" = "embed" ]; then
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
        echo "Usage for EMBEDDING:"
        echo "  ./test_local.sh [video_file] [transcript_file] [video_name]"
        echo "  ./test_local.sh embed [video_file] [transcript_file] [video_name]"
        echo ""
        echo "Or place video.mp4 and transcript.json in the video-search directory"
        echo "and run: ./test_local.sh"
        exit 1
    fi

    if [ -z "$TRANSCRIPT_FILE" ] || [ ! -f "$TRANSCRIPT_FILE" ]; then
        echo -e "${RED}Error: Transcript file not found!${NC}"
        echo ""
        echo "Usage for EMBEDDING:"
        echo "  ./test_local.sh [video_file] [transcript_file] [video_name]"
        echo "  ./test_local.sh embed [video_file] [transcript_file] [video_name]"
        echo ""
        echo "Or place video.mp4 and transcript.json in the video-search directory"
        echo "and run: ./test_local.sh"
        exit 1
    fi

    # Convert to absolute paths
    VIDEO_PATH=$(realpath "$VIDEO_FILE")
    TRANSCRIPT_PATH=$(realpath "$TRANSCRIPT_FILE")
fi

ENV_PATH=$(realpath ".env" 2>/dev/null || echo "")

# Check if Docker image exists, if not, build it (unless --no-build flag is set)
if ! docker images | grep -q "^${IMAGE_NAME}.*${IMAGE_TAG}"; then
    if [ "$SKIP_BUILD" = true ]; then
        echo -e "${RED}Error: Docker image ${IMAGE_NAME}:${IMAGE_TAG} not found and --no-build flag is set.${NC}"
        echo -e "${YELLOW}Please build the image first with: ./test_docker_build.sh${NC}"
        echo -e "${YELLOW}Or run without --no-build flag to build automatically.${NC}"
        exit 1
    fi
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
            if [ "$SKIP_BUILD" = true ]; then
                echo -e "${YELLOW}Warning: /local.py not found in image, but --no-build flag is set.${NC}"
                echo -e "${YELLOW}Will attempt to mount local.py from host if available.${NC}"
            else
                echo -e "${YELLOW}Warning: /local.py not found in image and local file not found.${NC}"
                echo -e "${YELLOW}Rebuilding image to include local.py...${NC}"
                echo ""
                ./test_docker_build.sh
                if [ $? -ne 0 ]; then
                    echo -e "${RED}Failed to build Docker image. Exiting.${NC}"
                    exit 1
                fi
                echo ""
            fi
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
if [ "$TASK_TYPE" = "search" ]; then
    echo -e "${GREEN}Testing SEARCH with Docker${NC}"
elif [ "$TASK_TYPE" = "analyse" ]; then
    echo -e "${GREEN}Testing ANALYSE with Docker${NC}"
else
    echo -e "${GREEN}Testing EMBEDDING with Docker${NC}"
fi
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Docker Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Task Type: $TASK_TYPE"
if [ "$TASK_TYPE" = "search" ]; then
    echo "Search Type: $SEARCH_TYPE"
    echo "Query: $QUERY"
    echo "Top K: $TOP_K"
elif [ "$TASK_TYPE" = "analyse" ]; then
    echo "Attachment ID: $ATTACHMENT_ID"
    echo "Query: $QUERY"
else
    echo "Video file: $VIDEO_PATH"
    echo "Transcript file: $TRANSCRIPT_PATH"
    echo "Video name: $VIDEO_NAME"
fi
if [ -n "$ENV_PATH" ]; then
    echo ".env file: $ENV_PATH"
fi
echo ""

# Get local.py and handler.py paths
LOCAL_PY_PATH=$(realpath "gpu/local.py" 2>/dev/null || echo "")
HANDLER_PY_PATH=$(realpath "gpu/handler.py" 2>/dev/null || echo "")
SRC_DIR_PATH=$(realpath "src" 2>/dev/null || echo "")

# Prepare Docker volume mounts
VOLUME_MOUNTS=()

# Only mount video/transcript files for embedding mode
if [ "$TASK_TYPE" = "embedding" ] || [ "$TASK_TYPE" = "embed" ]; then
    VOLUME_MOUNTS+=("-v" "$VIDEO_PATH:/mnt/video:ro")
    VOLUME_MOUNTS+=("-v" "$TRANSCRIPT_PATH:/mnt/transcript:ro")
fi

# Mount local.py if it exists (for development/testing without rebuilding)
if [ -n "$LOCAL_PY_PATH" ] && [ -f "$LOCAL_PY_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$LOCAL_PY_PATH:/local.py:ro")
fi

# Mount handler.py if it exists (to use latest version without rebuilding)
if [ -n "$HANDLER_PY_PATH" ] && [ -f "$HANDLER_PY_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$HANDLER_PY_PATH:/handler.py:ro")
fi

# Mount src directory if it exists (to use latest source code without rebuilding)
if [ -n "$SRC_DIR_PATH" ] && [ -d "$SRC_DIR_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$SRC_DIR_PATH:/src:ro")
fi

# Mount .env file if it exists
if [ -n "$ENV_PATH" ] && [ -f "$ENV_PATH" ]; then
    VOLUME_MOUNTS+=("-v" "$ENV_PATH:/.env:ro")
fi

# For search mode, check if query is a local file path (not a URL) and mount it
SEARCH_FILE_PATH=""
SEARCH_FILE_MOUNT_PATH=""
if [ "$TASK_TYPE" = "search" ]; then
    # Check if query is a local file (not starting with http:// or https://)
    if [ -n "$QUERY" ] && [[ ! "$QUERY" =~ ^https?:// ]] && [ -f "$QUERY" ]; then
        # It's a local file, mount it
        SEARCH_FILE_PATH=$(realpath "$QUERY")
        # Determine file extension to set appropriate mount path
        FILE_EXT="${SEARCH_FILE_PATH##*.}"
        if [ "$SEARCH_TYPE" = "image" ]; then
            SEARCH_FILE_MOUNT_PATH="/mnt/search_image.$FILE_EXT"
        elif [ "$SEARCH_TYPE" = "audio" ]; then
            SEARCH_FILE_MOUNT_PATH="/mnt/search_audio.$FILE_EXT"
        else
            SEARCH_FILE_MOUNT_PATH="/mnt/search_file.$FILE_EXT"
        fi
        VOLUME_MOUNTS+=("-v" "$SEARCH_FILE_PATH:$SEARCH_FILE_MOUNT_PATH:ro")
        echo "Mounting local file: $SEARCH_FILE_PATH -> $SEARCH_FILE_MOUNT_PATH"
    fi
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

# Add environment variables
DOCKER_CMD+=("-e" "RUNTIME_ENVIRONMENT=local")

# Add image
DOCKER_CMD+=("${IMAGE_NAME}:${IMAGE_TAG}")

# Build command based on task type
if [ "$TASK_TYPE" = "search" ]; then
    # Search mode: python3.11 /local.py search <search_type> <query> [top_k]
    # Use mounted path if local file was mounted, otherwise use original query (URL or path)
    SEARCH_QUERY="$QUERY"
    if [ -n "$SEARCH_FILE_MOUNT_PATH" ]; then
        SEARCH_QUERY="$SEARCH_FILE_MOUNT_PATH"
    fi
    DOCKER_CMD+=("python3.11" "/local.py" "search" "$SEARCH_TYPE" "$SEARCH_QUERY" "$TOP_K")
elif [ "$TASK_TYPE" = "analyse" ]; then
    # Analyse mode: python3.11 /local.py analyse <attachment_id> <query>
    DOCKER_CMD+=("python3.11" "/local.py" "analyse" "$ATTACHMENT_ID" "$QUERY")
else
    # Embedding mode: python3.11 /local.py /mnt/video /mnt/transcript <video_name>
    DOCKER_CMD+=("python3.11" "/local.py" "/mnt/video" "/mnt/transcript" "$VIDEO_NAME")
fi

# Execute the command
"${DOCKER_CMD[@]}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Container execution completed${NC}"
echo -e "${GREEN}========================================${NC}"
