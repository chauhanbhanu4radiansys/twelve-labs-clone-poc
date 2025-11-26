# Configuration Placeholders

This document lists all places where you need to update credentials.

## Pinecone Configuration

**File:** `src/embedding/config.py`  
**Lines:** 7-13

```python
PINECONE_CONFIG = {
    "api_key": "YOUR_PINECONE_API_KEY",  # PLACEHOLDER: Update with your Pinecone API key
    "video_index_name": "video-search",  # PLACEHOLDER: Update if different
    "audio_index_name": "audio-search",  # PLACEHOLDER: Update if different
    "text_index_name": "text-search",    # PLACEHOLDER: Update if different
    "desc_index_name": "desc-search"      # PLACEHOLDER: Update if different
}
```

## Summary

All placeholders are located in a single file:
- **`src/embedding/config.py`** - Contains Pinecone configuration

**Note:** MinIO is no longer used. Video and transcript are accessed via pre-signed S3 URLs.

## Usage

The pipeline accepts pre-signed S3 URLs for both video and transcript:

```python
from src.embedding import process_video

process_video(
    videoPathURL="https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",
    transcriptPathURL="https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",
    video_name="my_video.mp4",
    video_id="unique-id",  # Optional
    status_callback=print  # Optional
)
```

## Placeholder Locations

1. **Pinecone API Key** - `src/embedding/config.py` line 10
2. **Pinecone Index Names** - `src/embedding/config.py` lines 11-14
