# Quick Start Guide

## Step 1: Install Dependencies

```bash
# Install system dependency
brew install ffmpeg  # macOS
# OR
sudo apt-get install ffmpeg  # Linux

# Install Python packages
pip install torch torchvision torchaudio
pip install git+https://github.com/facebookresearch/ImageBind.git
pip install transformers opencv-python scenedetect pydub pinecone pillow requests numpy
```

## Step 2: Configure Pinecone

Edit `src/embedding/config.py`:

```python
PINECONE_CONFIG = {
    "api_key": "your-actual-api-key-here",  # ← Update this
    "video_index_name": "video-search",
    "audio_index_name": "audio-search",
    "text_index_name": "text-search",
    "desc_index_name": "desc-search"
}
```

**Create Pinecone Indexes:**
- Go to Pinecone dashboard
- Create 4 indexes with:
  - Dimension: `1024`
  - Metric: `cosine`
  - Names: `video-search`, `audio-search`, `text-search`, `desc-search`

## Step 3: Run

```python
from src.embedding import process_video

process_video(
    videoPathURL="https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",
    transcriptPathURL="https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",
    video_name="my_video.mp4"
)
```

Or use the example script:
```bash
# Edit src/embedding/run_example.py with your URLs
python src/embedding/run_example.py
```

## What's Missing?

Before running, ensure you have:

✅ **Python 3.10+**  
✅ **ffmpeg installed**  
✅ **All Python dependencies installed**  
✅ **Pinecone API key configured**  
✅ **Pinecone indexes created**  
✅ **Pre-signed S3 URLs for video and transcript**

## Check Installation

```python
# Test imports
python -c "from src.embedding import process_video; print('✓ Imports OK')"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')"
python -c "from pinecone import Pinecone; print('✓ Pinecone OK')"
```

