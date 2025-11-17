# Quick Start Guide

## Installation

### Prerequisites
- Python 3.10 or 3.11
- CUDA-capable GPU (recommended) or CPU
- ffmpeg installed on your system

### Step 1: Create and activate a conda environment (recommended)

```bash
conda create --name videoprism python=3.10 -y
conda activate videoprism
```

### Step 2: Install dependencies

**Option A: Using the installation script (Recommended)**
```bash
bash install.sh
```

**Option B: Manual installation**
```bash
# 1. Install PyTorch first
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU-only version:
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# 2. Install remaining dependencies
pip install -r requirements.txt
```

### Step 3: Configure secrets

Create `.streamlit/secrets.toml` with your credentials:
- MinIO/S3 configuration
- MongoDB URI
- Pinecone API key
- OpenAI API key (optional)

### Step 4: Run the application

```bash
streamlit run app_rag_chat.py
```

## Files

- `app_rag_chat.py` - Main application file
- `requirements.txt` - All Python dependencies (standalone)
- `install.sh` - Automated installation script
- `README.md` - Full documentation

