#!/bin/bash

# Test Docker Build Script
# This script builds the Docker image from the Dockerfile in the gpu directory

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="video-search-gpu"
IMAGE_TAG="latest"
DOCKERFILE_PATH="./gpu/Dockerfile"
BUILD_CONTEXT="."

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker Image Build Test Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Image Name: $IMAGE_NAME"
echo "  Image Tag: $IMAGE_TAG"
echo "  Dockerfile: $DOCKERFILE_PATH"
echo "  Build Context: $BUILD_CONTEXT"
echo ""

# Function to check if Docker is accessible
check_docker() {
    docker info > /dev/null 2>&1
}

# Check if Docker is running
echo -e "${YELLOW}Checking Docker daemon...${NC}"
if ! check_docker; then
    echo -e "${YELLOW}Docker daemon is not running or not accessible.${NC}"
    
    # Try to start Docker on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "${YELLOW}Attempting to start Docker daemon...${NC}"
        if sudo systemctl start docker 2>/dev/null; then
            echo -e "${GREEN}✓ Docker daemon start command executed${NC}"
            
            # Wait for Docker to fully initialize with retries
            echo -e "${YELLOW}Waiting for Docker to initialize...${NC}"
            MAX_RETRIES=10
            RETRY_COUNT=0
            while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
                sleep 1
                if check_docker; then
                    echo -e "${GREEN}✓ Docker daemon is running and responding${NC}"
                    echo ""
                    break
                fi
                RETRY_COUNT=$((RETRY_COUNT + 1))
                echo -n "."
            done
            echo ""
            
            if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
                echo -e "${RED}Error: Docker daemon started but is not responding after ${MAX_RETRIES} seconds.${NC}"
                echo ""
                echo "This might be a permission issue. Try:"
                echo "  1. Add your user to the docker group: sudo usermod -aG docker \$USER"
                echo "  2. Log out and log back in, or run: newgrp docker"
                echo "  3. Check Docker status: sudo systemctl status docker"
                echo ""
                exit 1
            fi
        else
            echo -e "${RED}Error: Could not start Docker daemon automatically.${NC}"
            echo ""
            echo "Please start Docker manually and try again:"
            echo "  sudo systemctl start docker"
            echo ""
            echo "Or check Docker status:"
            echo "  sudo systemctl status docker"
            echo ""
            exit 1
        fi
    else
        echo -e "${RED}Error: Docker daemon is not running.${NC}"
        echo ""
        echo "Please start Docker and try again:"
        echo "  - On Linux: sudo systemctl start docker"
        echo "  - On macOS: Open Docker Desktop application"
        echo "  - On Windows: Start Docker Desktop"
        echo ""
        exit 1
    fi
else
    echo -e "${GREEN}✓ Docker daemon is running${NC}"
fi
echo ""

echo -e "${GREEN}Starting Docker build...${NC}"
echo ""

# Build the Docker image
# Using the current directory as build context since Dockerfile references ../src
docker build \
    -f "$DOCKERFILE_PATH" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$BUILD_CONTEXT"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To run the container:"
    echo "  docker run --gpus all ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To list the image:"
    echo "  docker images | grep ${IMAGE_NAME}"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Docker build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

