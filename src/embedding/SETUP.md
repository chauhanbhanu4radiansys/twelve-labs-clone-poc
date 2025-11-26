# Setup and Run Guide

## Prerequisites

### 1. System Dependencies

**ffmpeg** (required for video/audio processing):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Verify installation
ffmpeg -version
```

### 2. Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use conda
conda create -n videoprism python=3.10
conda activate videoprism
```

### 3. Install Python Dependencies

```bash
# Install core dependencies
pip install torch torchvision torchaudio

# Install ImageBind (from GitHub)
pip install git+https://github.com/facebookresearch/ImageBind.git

# Install other dependencies
pip install transformers>=4.35.2
pip install opencv-python>=4.8.0
pip install scenedetect[opencv]>=0.6.2
pip install pydub>=0.25.1
pip install pinecone>=3.0.0
pip install pillow>=9.0.0
pip install requests>=2.31.0
pip install numpy<2.0
```

Or install from the requirements file:
```bash
pip install -r src/embedding/requirements.txt
```

### 4. ImageBind Model Checkpoint (Optional but Recommended)

The model will auto-download if not found, but you can pre-download:

```bash
# Create checkpoints directory
mkdir -p .checkpoints

# Download ImageBind huge model (2.5GB)
# The model will auto-download on first run if not present
```

**Checkpoint location:** `.checkpoints/imagebind_huge.pth`

## Configuration

### Update Pinecone Credentials

**File:** `src/embedding/config.py`

```python
PINECONE_CONFIG = {
    "api_key": "YOUR_PINECONE_API_KEY",  # ← Update this
    "video_index_name": "video-search",
    "audio_index_name": "audio-search",
    "text_index_name": "text-search",
    "desc_index_name": "desc-search"
}
```

**Required Pinecone Indexes:**
- Create 4 indexes in Pinecone with:
  - Dimension: `1024`
  - Metric: `cosine`
  - Names: `video-search`, `audio-search`, `text-search`, `desc-search`

## Running the Pipeline

### Method 1: Using the Example Script

```bash
# Edit run_example.py and update the URLs
python src/embedding/run_example.py
```

### Method 2: Python Script

```python
from src.embedding import process_video

process_video(
    videoPathURL="https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",
    transcriptPathURL="https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",
    video_name="my_video.mp4",
    video_id="unique-id",  # Optional
    status_callback=print
)
```

### Method 3: Command Line (if you create a CLI)

```bash
python -m src.embedding.run_example
```

## What You Need

### Required Inputs:
1. **Pre-signed S3 URL for video** - Public URL to video file
2. **Pre-signed S3 URL for transcript** - Public URL to transcript JSON file
3. **Pinecone API key** - For uploading embeddings
4. **Pinecone indexes** - 4 indexes must exist

### Transcript Format:

The transcript JSON should be in one of these formats:

**Format 1 (List of segments):**
```json
[
    {"start": 0.0, "end": 5.2, "text": "Hello, welcome..."},
    {"start": 5.2, "end": 10.5, "text": "Today we'll learn..."}
]
```

**Format 2 (Object with segments):**
```json
{
    "segments": [
        {"start": 0.0, "end": 5.2, "text": "Hello, welcome..."}
    ]
}
```

## Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError: No module named 'imagebind'"**
   ```bash
   pip install git+https://github.com/facebookresearch/ImageBind.git
   ```

2. **"ffmpeg not found"**
   - Install ffmpeg system-wide (see Prerequisites)

3. **"CUDA out of memory"**
   - The code will automatically fallback to CPU
   - Or reduce BATCH_SIZE in config.py

4. **"Pinecone index not found"**
   - Create the indexes in Pinecone dashboard
   - Check index names match config.py

5. **"Failed to download from URL"**
   - Verify pre-signed URLs are valid and not expired
   - Check network connectivity

## Expected Output

The pipeline will:
1. Download video and transcript from URLs
2. Process video (scene detection, embeddings)
3. Upload embeddings to Pinecone
4. Print status updates and final statistics

Example output:
```
[STATUS] Initializing Pinecone clients...
[STATUS] Downloading video from URL...
[STATUS] Video downloaded to temporary file: /tmp/tmpXXXXXX.mp4
[STATUS] Downloading transcript from URL...
[STATUS] Loading models...
[STATUS] Extracting audio from video...
[STATUS] Splitting video into chunks...
[STATUS] Processing 1 video chunk(s) in parallel...
[STATUS] Detected 25 scenes. Starting parallel processing...
[STATUS] Completed processing chunk 1/1...
[STATUS] Waiting for Pinecone uploads to complete...
[STATUS] Processing complete! 25 video, 25 audio, 25 text, 25 description embeddings uploaded.
```

