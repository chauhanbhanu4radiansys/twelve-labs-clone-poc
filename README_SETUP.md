# Setup Instructions

## Quick Setup

Run the setup script to install all dependencies:

```bash
./setup.sh
```

This script will:
1. ✅ Check Python and pip availability
2. ✅ Verify ffmpeg is installed (doesn't install it)
3. ✅ Upgrade pip
4. ✅ Install PyTorch
5. ✅ Install ImageBind from GitHub
6. ✅ Install all Python packages from `requirements.txt`
7. ✅ Verify installation

## Manual Setup

If you prefer to install manually:

```bash
# 1. Install PyTorch
pip install torch torchvision torchaudio

# 2. Install ImageBind from GitHub
pip install git+https://github.com/facebookresearch/ImageBind.git

# 3. Install other dependencies
pip install -r requirements.txt
```

## Files

- **`requirements.txt`** - All Python package dependencies
- **`setup.sh`** - Automated setup script (installs ImageBind + calls requirements.txt)

## Prerequisites

- ✅ Python 3.10+
- ✅ pip
- ✅ ffmpeg (already installed, not installed by script)
- ✅ git (for ImageBind installation)

## After Setup

1. Update Pinecone credentials in `src/embedding/config.py`
2. Create Pinecone indexes (video-search, audio-search, text-search, desc-search)
3. Run: `python src/embedding/run_example.py`

