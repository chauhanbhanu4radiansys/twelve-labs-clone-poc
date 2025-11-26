# How to Run the Embedding Pipeline

## Quick Answer

### 1. Install Dependencies
```bash
# System dependency
brew install ffmpeg  # macOS
# OR sudo apt-get install ffmpeg  # Linux

# Python packages
pip install -r requirements.txt
pip install requests  # If not already installed
```

### 2. Configure Pinecone
Edit `src/embedding/config.py` and update:
- `api_key`: Your Pinecone API key
- Index names (if different from defaults)

### 3. Create Pinecone Indexes
In Pinecone dashboard, create 4 indexes:
- `video-search` (dimension: 1024, metric: cosine)
- `audio-search` (dimension: 1024, metric: cosine)
- `text-search` (dimension: 1024, metric: cosine)
- `desc-search` (dimension: 1024, metric: cosine)

### 4. Run
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
python src/embedding/run_example.py
```

---

## Detailed Setup

### Missing Dependencies

The code requires these that might not be in your environment:

1. **System:**
   - ✅ `ffmpeg` - Video/audio processing (install system-wide)

2. **Python Packages:**
   - ✅ `requests` - HTTP downloads (added to requirements.txt)
   - ✅ `torch` - PyTorch for ML models
   - ✅ `imagebind` - ImageBind model (from GitHub)
   - ✅ `transformers` - BLIP model
   - ✅ `opencv-python` - Video processing
   - ✅ `scenedetect` - Scene detection
   - ✅ `pydub` - Audio processing
   - ✅ `pinecone` - Vector database client
   - ✅ `pillow` - Image processing

### Installation Commands

```bash
# 1. Install ffmpeg
brew install ffmpeg  # macOS
# OR
sudo apt-get install ffmpeg  # Ubuntu/Debian

# 2. Install Python packages
pip install torch torchvision torchaudio
pip install git+https://github.com/facebookresearch/ImageBind.git
pip install transformers>=4.35.2
pip install opencv-python>=4.8.0
pip install scenedetect[opencv]>=0.6.2
pip install pydub>=0.25.1
pip install pinecone>=3.0.0
pip install pillow>=9.0.0
pip install requests>=2.31.0
pip install numpy<2.0

# OR install from requirements
pip install -r requirements.txt
pip install requests  # Ensure requests is installed
```

### Configuration Checklist

Before running, ensure:

- [ ] **Pinecone API key** set in `src/embedding/config.py`
- [ ] **4 Pinecone indexes created** (video-search, audio-search, text-search, desc-search)
- [ ] **Pre-signed S3 URLs** ready for video and transcript
- [ ] **ffmpeg installed** and accessible in PATH
- [ ] **All Python packages installed**

### Running the Code

**Option 1: Direct Python Import**
```python
from src.embedding import process_video

def status_callback(msg):
    print(f"[STATUS] {msg}")

process_video(
    videoPathURL="https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",
    transcriptPathURL="https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",
    video_name="my_video.mp4",
    video_id="unique-id",  # Optional
    status_callback=status_callback
)
```

**Option 2: Example Script**
```bash
# Edit src/embedding/run_example.py with your URLs
python src/embedding/run_example.py
```

**Option 3: Command Line (if you create a CLI wrapper)**
```bash
python -m src.embedding.run_example
```

### Expected Output

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

### Troubleshooting

**"ModuleNotFoundError: No module named 'requests'"**
```bash
pip install requests
```

**"ffmpeg not found"**
```bash
brew install ffmpeg  # macOS
# OR
sudo apt-get install ffmpeg  # Linux
```

**"Pinecone index not found"**
- Create indexes in Pinecone dashboard
- Check index names match `config.py`

**"Failed to download from URL"**
- Verify pre-signed URLs are valid and not expired
- Check network connectivity

**"CUDA out of memory"**
- Code auto-fallback to CPU
- Or reduce `BATCH_SIZE` in `config.py`

### What the Pipeline Does

1. ✅ Downloads video from pre-signed S3 URL
2. ✅ Downloads transcript from pre-signed S3 URL
3. ✅ Loads ML models (ImageBind, BLIP)
4. ✅ Extracts audio from video
5. ✅ Detects scenes using PySceneDetect
6. ✅ Generates embeddings (visual, audio, text, description)
7. ✅ Uploads embeddings to Pinecone
8. ✅ Cleans up temporary files

### Files Created

- `src/embedding/run_example.py` - Example runner script
- `src/embedding/requirements.txt` - Dependencies list
- `src/embedding/SETUP.md` - Detailed setup guide
- `src/embedding/QUICKSTART.md` - Quick reference

