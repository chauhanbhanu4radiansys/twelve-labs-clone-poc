#!/bin/bash

# Script to install NVIDIA Container Toolkit for Docker GPU support
# Run with: sudo ./install_nvidia_toolkit.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installing NVIDIA Container Toolkit${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Get distribution
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo -e "${YELLOW}Detected distribution: $distribution${NC}"
echo ""

# Step 1: Add GPG key
echo -e "${YELLOW}Step 1/5: Adding NVIDIA GPG key...${NC}"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo -e "${GREEN}✓ GPG key added${NC}"
echo ""

# Step 2: Add repository
echo -e "${YELLOW}Step 2/5: Adding NVIDIA repository...${NC}"
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
echo -e "${GREEN}✓ Repository added${NC}"
echo ""

# Step 3: Update package list
echo -e "${YELLOW}Step 3/5: Updating package list...${NC}"
apt-get update -y
echo -e "${GREEN}✓ Package list updated${NC}"
echo ""

# Step 4: Install NVIDIA Container Toolkit
echo -e "${YELLOW}Step 4/5: Installing nvidia-container-toolkit...${NC}"
apt-get install -y nvidia-container-toolkit
echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
echo ""

# Step 5: Restart Docker
echo -e "${YELLOW}Step 5/5: Restarting Docker service...${NC}"
systemctl restart docker
echo -e "${GREEN}✓ Docker restarted${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Testing GPU access...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU access verified!${NC}"
    echo ""
    echo -e "${GREEN}You can now run: ./test_local.sh${NC}"
else
    echo -e "${RED}✗ GPU access test failed${NC}"
    echo -e "${YELLOW}Try running manually:${NC}"
    echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
fi

