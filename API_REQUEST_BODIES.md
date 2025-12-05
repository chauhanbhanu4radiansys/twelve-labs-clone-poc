# API Request Bodies Documentation

This document provides complete JSON request body examples for all task types supported by `handler.py`.

## Table of Contents
1. [EMBEDDING Task](#embedding-task)
2. [SEARCH Task](#search-task)
3. [ANALYSE Task](#analyse-task)
4. [Response Format](#response-format)

---

## EMBEDDING Task

Processes a video and generates embeddings for search.

### Request Body

```json
{
  "input": {
    "task_type": "embedding",
    "videoPathURL": "https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
    "transcriptPathURL": "https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
    "video_name": "my_video_name",
    "video_id": "optional_video_id_123",
    "id": "job_id_456",
    "tenant": "tenant_id_789",
    "snsTopicArn": "arn:aws:sns:us-east-1:123456789012:video-processing-topic",
    "duration": 10.0,
    "startTime": 0.0
  }
}
```

### Field Descriptions

#### Required Fields:
- **`videoPathURL`** (string): S3 pre-signed URL to the video file
  - Example: `"https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=..."`
  - Supports MP4, AVI, MOV, MKV formats
  
- **`transcriptPathURL`** (string): S3 pre-signed URL to the transcript JSON file
  - Example: `"https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=..."`
  - Expected format: JSON array of segments with `start`, `end`, `text` keys
  
- **`video_name`** (string): Name/identifier for the video
  - Example: `"TrainVsGiantPit.mp4"`

#### Optional Fields:
- **`task_type`** (string): Task type identifier
  - Default: `"embedding"` (if not provided)
  - Values: `"embedding"`
  
- **`video_id`** (string): Unique identifier for the video
  - If not provided, a UUID will be auto-generated
  - Example: `"69326bd6f206715c442b4c1c"`
  
- **`id`** (string): Job ID for tracking and notifications
  - Example: `"job-123-456"`
  
- **`tenant`** (string): Tenant ID for multi-tenant support
  - Example: `"tenant-abc-123"`
  
- **`snsTopicArn`** (string): AWS SNS topic ARN for success/failure notifications
  - Example: `"arn:aws:sns:us-east-1:123456789012:video-processing-topic"`
  - If provided, notifications will be sent on completion/failure
  
- **`duration`** (float): Video duration in seconds (for partial video processing)
  - Example: `10.0` (10 seconds)
  - Used with `startTime` for range downloads
  
- **`startTime`** (float): Start time in seconds (for partial video processing)
  - Example: `0.0` (start from beginning)
  - Used with `duration` for range downloads

### Minimal Request (Required Fields Only)

```json
{
  "input": {
    "videoPathURL": "https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",
    "transcriptPathURL": "https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",
    "video_name": "my_video"
  }
}
```

### Example with Partial Video Processing

```json
{
  "input": {
    "task_type": "embedding",
    "videoPathURL": "https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...",
    "transcriptPathURL": "https://s3.amazonaws.com/bucket/transcript.json?X-Amz-Algorithm=...",
    "video_name": "my_video",
    "duration": 30.0,
    "startTime": 60.0
  }
}
```
This will process only the segment from 60s to 90s.

---

## SEARCH Task

Searches across indexed videos using text, image, or audio queries.

### Field Descriptions

#### Required Fields:
- **`task_type`** (string): Must be `"search"`
- **`search_type`** (string): Type of search
  - Values: `"text"`, `"image"`, or `"audio"`
- **`query`** (string): Search query
  - For text: Search query string
  - For image/audio: S3 pre-signed URL or local file path

#### Optional Fields:
- **`top_k`** (integer): Number of results to return
  - Default: `8`
  - Range: 1-100 (recommended)
- **`filter`** (object): Metadata filter dictionary
  - Example: `{"video_name": "TrainVsGiantPit.mp4"}`
  - Filters results by metadata fields

---

### 1. Text Search

Searches using a text query across video frames, audio transcripts, and descriptions.

#### Request Body

```json
{
  "input": {
    "task_type": "search",
    "search_type": "text",
    "query": "a person running",
    "top_k": 8,
    "filter": {
      "video_name": "TrainVsGiantPit.mp4"
    }
  }
}
```

#### Minimal Request

```json
{
  "input": {
    "task_type": "search",
    "search_type": "text",
    "query": "a person running"
  }
}
```

#### Examples

**Simple text query:**
```json
{
  "input": {
    "task_type": "search",
    "search_type": "text",
    "query": "car crash",
    "top_k": 10
  }
}
```

**With metadata filter:**
```json
{
  "input": {
    "task_type": "search",
    "search_type": "text",
    "query": "where does the speaker mention earnings",
    "top_k": 5,
    "filter": {
      "video_name": "Q3_Earnings_Call.mp4"
    }
  }
}
```

---

### 2. Image Search

Searches using an image query. Supports S3 pre-signed URLs (production) or local file paths (testing).

#### Request Body (S3 Pre-signed URL - Production)

```json
{
  "input": {
    "task_type": "search",
    "search_type": "image",
    "query": "https://s3.amazonaws.com/bucket/search_image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
    "top_k": 10
  }
}
```

#### Request Body (Local File Path - Testing)

```json
{
  "input": {
    "task_type": "search",
    "search_type": "image",
    "query": "/mnt/search_image.png",
    "top_k": 8
  }
}
```

#### Minimal Request

```json
{
  "input": {
    "task_type": "search",
    "search_type": "image",
    "query": "https://s3.amazonaws.com/bucket/image.jpg?X-Amz-Algorithm=..."
  }
}
```

#### Supported Image Formats
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- Other formats supported by PIL/Pillow

#### Examples

**S3 URL with filter:**
```json
{
  "input": {
    "task_type": "search",
    "search_type": "image",
    "query": "https://s3.amazonaws.com/bucket/car_image.png?X-Amz-Algorithm=...",
    "top_k": 15,
    "filter": {
      "video_name": "TrainVsGiantPit.mp4"
    }
  }
}
```

**Local file path (for Docker testing):**
```json
{
  "input": {
    "task_type": "search",
    "search_type": "image",
    "query": "/mnt/search_image.png",
    "top_k": 8
  }
}
```

---

### 3. Audio Search

Searches using an audio query. Supports S3 pre-signed URLs (production) or local file paths (testing).

#### Request Body (S3 Pre-signed URL - Production)

```json
{
  "input": {
    "task_type": "search",
    "search_type": "audio",
    "query": "https://s3.amazonaws.com/bucket/search_audio.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
    "top_k": 8
  }
}
```

#### Request Body (Local File Path - Testing)

```json
{
  "input": {
    "task_type": "search",
    "search_type": "audio",
    "query": "/mnt/search_audio.wav",
    "top_k": 8
  }
}
```

#### Minimal Request

```json
{
  "input": {
    "task_type": "search",
    "search_type": "audio",
    "query": "https://s3.amazonaws.com/bucket/audio.wav?X-Amz-Algorithm=..."
  }
}
```

#### Supported Audio Formats
- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- Other formats supported by ffmpeg

#### Examples

**S3 URL:**
```json
{
  "input": {
    "task_type": "search",
    "search_type": "audio",
    "query": "https://s3.amazonaws.com/bucket/sound_clip.mp3?X-Amz-Algorithm=...",
    "top_k": 12
  }
}
```

**Local file path (for Docker testing):**
```json
{
  "input": {
    "task_type": "search",
    "search_type": "audio",
    "query": "/mnt/search_audio.wav",
    "top_k": 8
  }
}
```

---

## ANALYSE Task

Analyzes video content using LLM-based analysis. Fetches video data from Pinecone and generates analysis with clips.

### Request Body

```json
{
  "input": {
    "task_type": "analyse",
    "attachment_id": "69326bd6f206715c442b4c1c",
    "query": "What are the key moments in this video?"
  }
}
```

### Field Descriptions

#### Required Fields:
- **`task_type`** (string): Must be `"analyse"` or `"analyze"` (both accepted)
- **`attachment_id`** (string): Video document ID
  - Used as `video_doc_id` in Pinecone to identify the video
  - Example: `"69326bd6f206715c442b4c1c"`
  
- **`query`** (string): Analysis prompt/question
  - Example: `"What are the key moments in this video?"`
  - Example: `"Summarize this video"`
  - Example: `"Generate hashtags and topics"`

### Examples

**Holistic analysis (summary):**
```json
{
  "input": {
    "task_type": "analyse",
    "attachment_id": "69326bd6f206715c442b4c1c",
    "query": "Summarize this video in 3-5 key points"
  }
}
```

**Specific question:**
```json
{
  "input": {
    "task_type": "analyse",
    "attachment_id": "69326bd6f206715c442b4c1c",
    "query": "Where does the speaker discuss the Q3 earnings?"
  }
}
```

**Content generation:**
```json
{
  "input": {
    "task_type": "analyse",
    "attachment_id": "69326bd6f206715c442b4c1c",
    "query": "Generate hashtags and topics for this video"
  }
}
```

**Key moments extraction:**
```json
{
  "input": {
    "task_type": "analyse",
    "attachment_id": "69326bd6f206715c442b4c1c",
    "query": "What are the key moments in this video?"
  }
}
```

---

## Response Format

All tasks return responses in a consistent format:

### Success Response Structure

```json
{
  "statusCode": 200,
  "body": "{\"status\": \"success\", ...}"
}
```

The `body` field contains a JSON string that can be parsed. The structure varies by task type:

---

### EMBEDDING Response

```json
{
  "status": "success",
  "message": "Video embeddings processed successfully",
  "video_name": "my_video",
  "video_id": "69326bd6f206715c442b4c1c"
}
```

---

### SEARCH Response

```json
{
  "status": "success",
  "search_type": "image",
  "results_count": 13,
  "original_results_count": 15,
  "scene_merging": true,
  "merge_gap_seconds": 3,
  "results": [
    {
      "score": 1.001,
      "category": "HIGH",
      "video_name": "TrainVsGiantPit.mp4",
      "video_id": "69326bd6f206715c442b4c1c",
      "video_doc_id": "69326bd6f206715c442b4c1c",
      "start_time": 271.53793333333334,
      "end_time": 275.10816666666665,
      "start_timestamp": "00:04:31",
      "end_timestamp": "00:04:35",
      "time_range": "00:04:31 - 00:04:35",
      "transcript": "Yes.",
      "description": "arafed car on its side on the side of the road",
      "source": "desc, video",
      "scene_uuid": "c25b23bd-439d-4a16-9b02-c130f57085a1",
      "scene_index": 278,
      "metadata": {
        "description": "arafed car on its side on the side of the road",
        "end_time": 275.10816666666665,
        "scene_index": 278,
        "scene_uuid": "c25b23bd-439d-4a16-9b02-c130f57085a1",
        "start_time": 271.53793333333334,
        "transcript": "Yes.",
        "video_doc_id": "69326bd6f206715c442b4c1c",
        "video_name": "TrainVsGiantPit.mp4",
        "source": "desc, video"
      }
    }
  ]
}
```

#### Search Result Fields:
- **`score`** (float): Similarity score (0.0 to 1.0+)
- **`category`** (string): Score category - `"HIGH"` (≥0.5), `"MEDIUM"` (≥0.3), `"LOW"` (<0.3)
- **`video_name`** (string): Name of the video
- **`video_id`** (string): Video identifier
- **`video_doc_id`** (string): Video document ID in Pinecone
- **`start_time`** (float): Start time in seconds
- **`end_time`** (float): End time in seconds
- **`start_timestamp`** (string): Formatted start time (HH:MM:SS)
- **`end_timestamp`** (string): Formatted end time (HH:MM:SS)
- **`time_range`** (string): Display format "HH:MM:SS - HH:MM:SS"
- **`transcript`** (string): Transcript text for this scene
- **`description`** (string): BLIP-generated description
- **`source`** (string): Which modalities matched (e.g., "video, desc", "text, audio")
- **`scene_uuid`** (string): Unique scene identifier
- **`scene_index`** (integer): Scene index in the video
- **`metadata`** (object): Full metadata object

---

### ANALYSE Response

```json
{
  "status": "success",
  "attachment_id": "69326bd6f206715c442b4c1c",
  "query": "What are the key moments in this video?",
  "video_duration": 600.5,
  "transcript_segments_count": 245,
  "analysis": {
    "analysis_text": "This video contains several key moments...",
    "clips": [
      {
        "description": "The speaker discusses Q3 earnings and future projections.",
        "start_time": 125
      },
      {
        "description": "A major announcement about product launch.",
        "start_time": 340
      }
    ]
  }
}
```

#### Analysis Result Fields:
- **`analysis_text`** (string): Full analysis text from LLM
- **`clips`** (array): Array of clip objects
  - **`description`** (string): Description of the clip
  - **`start_time`** (integer): Start time in seconds

---

### Error Response

```json
{
  "statusCode": 500,
  "body": "{\"status\": \"error\", \"error\": \"Error message here\"}"
}
```

#### Common Error Scenarios:

**Missing required parameter:**
```json
{
  "statusCode": 500,
  "body": "{\"status\": \"error\", \"error\": \"Missing required parameter: query\"}"
}
```

**Invalid search type:**
```json
{
  "statusCode": 500,
  "body": "{\"status\": \"error\", \"error\": \"Invalid search_type: invalid_type. Must be 'text', 'image', or 'audio'.\"}"
}
```

**File not found:**
```json
{
  "statusCode": 500,
  "body": "{\"status\": \"error\", \"error\": \"Image file not found: /path/to/file.png\"}"
}
```

**Download failure:**
```json
{
  "statusCode": 500,
  "body": "{\"status\": \"error\", \"error\": \"Failed to download image from URL: Connection timeout\"}"
}
```

---

## Notes

### URL Handling
- **S3 Pre-signed URLs**: Use for production. Must include query parameters (`?X-Amz-Algorithm=...`)
- **Local File Paths**: Use for testing in Docker containers (e.g., `/mnt/search_image.png`)
- **HTTP/HTTPS URLs**: Also supported for image/audio search

### Scene Merging
- Search results automatically merge overlapping or nearby scenes (within 3 seconds)
- This is indicated by `"scene_merging": true` in the response
- `"merge_gap_seconds": 3` shows the gap threshold used

### Score Categories
- **HIGH**: Score ≥ 0.5 (strong match)
- **MEDIUM**: Score ≥ 0.3 and < 0.5 (moderate match)
- **LOW**: Score < 0.3 (weak match)

### Query Intent Detection (ANALYSE)
The analyse task automatically detects query intent:
- **SPECIFIC_QUESTION**: Uses RAG-based search for targeted answers
- **HOLISTIC_REQUEST**: Uses full transcript for comprehensive analysis

---

## Testing

### Using test_local.sh

**Embedding:**
```bash
./test_local.sh video.mp4 transcript.json "my_video"
```

**Text Search:**
```bash
./test_local.sh search text "a person running" 10
```

**Image Search (local file):**
```bash
./test_local.sh search image "im2.png" 8
```

**Image Search (URL):**
```bash
./test_local.sh search image "https://example.com/image.jpg" 8
```

**Audio Search:**
```bash
./test_local.sh search audio "audio.wav" 8
```

**Analyse:**
```bash
./test_local.sh analyse 69326bd6f206715c442b4c1c "What are the key moments?"
```

---

## Production Usage

### Example: Calling handler.py directly

```python
import json
from handler import handler

# Search request
search_job = {
    "input": {
        "task_type": "search",
        "search_type": "text",
        "query": "a person running",
        "top_k": 8
    }
}

result = handler(search_job)
print(json.loads(result['body']))
```

### Example: RunPod Serverless

The handler automatically starts RunPod serverless endpoint when `runpod` is available:

```python
# handler.py automatically handles this:
if runpod is not None:
    runpod.serverless.start({"handler": handler})
```

RunPod will call `handler()` with the job payload directly.

---

## Version Information

- **Last Updated**: 2024
- **Handler Version**: Compatible with `handler.py` and `local.py`
- **API Version**: 1.0

