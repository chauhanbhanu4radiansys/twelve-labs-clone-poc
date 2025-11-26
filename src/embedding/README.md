# Embedding Pipeline

This module implements the complete embedding generation pipeline for video processing.

## Structure

```
embedding/
├── __init__.py          # Main export: process_video()
├── config.py            # Configuration with placeholders
├── models.py            # Model loading (ImageBind, BLIP)
├── processing.py        # Video processing (scene detection, audio extraction)
├── embeddings.py        # Embedding generation
├── upload.py            # Pinecone upload manager
├── download.py          # Download utilities for pre-signed S3 URLs
└── pipeline.py          # Main orchestration pipeline
```

## Usage

```python
from src.embedding import process_video

# Process a video using pre-signed S3 URLs
process_video(
    videoPathURL="https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",  # Pre-signed S3 URL
    transcriptPathURL="https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",  # Pre-signed S3 URL
    video_name="my_video.mp4",
    video_id="unique-id-123",  # Optional, auto-generated if None
    status_callback=print  # Optional callback for status updates
)
```

## Pipeline Flow

1. **Initialize Pinecone** - Connect to Pinecone indexes
2. **Download Video** - Download video from pre-signed S3 URL to temporary file
3. **Download Transcript** - Download and parse transcript JSON from pre-signed S3 URL
4. **Load Models** - ImageBind, BLIP (Whisper not needed)
5. **Extract Audio** - Extract audio from downloaded video for scene audio slicing
6. **Split into Chunks** - If video > 10 minutes
7. **Process Chunks in Parallel**:
   - Detect scenes (PySceneDetect)
   - Extract keyframes, audio segments, match transcripts
   - Generate embeddings (ImageBind)
   - Upload to Pinecone (background workers)
8. **Cleanup** - Remove temporary files (downloaded video, audio, chunks)

## Configuration

Update placeholders in `config.py`:
- Pinecone API key and index names

**Note:** MinIO is no longer used. All data (video and transcript) comes from pre-signed S3 URLs.

See `../PLACEHOLDERS.md` for details.

## Output

The pipeline creates embeddings in 4 Pinecone indexes:
- `video-search` - Visual embeddings from keyframes
- `audio-search` - Audio embeddings from scene audio
- `text-search` - Text embeddings from matched transcript segments
- `desc-search` - Text embeddings from BLIP captions

Each embedding includes metadata:
- `video_name`, `video_id`, `scene_index`
- `start_time`, `end_time`
- `transcript`, `description`
- `scene_uuid`

