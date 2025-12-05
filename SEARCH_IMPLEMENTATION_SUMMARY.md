# Search Implementation Summary

## Overview
The codebase has been updated to support three types of search using two inputs:
- `search_type`: "text", "image", or "audio"
- `search_query`: Text string for text search, URL for image/audio search

## Files Already Updated (Before This Session)

### 1. `src/retrieval/search.py`
**Status:** ✅ Already Complete

**Key Function:** `search_videos()`
```python
def search_videos(
    search_type: str,      # 'text', 'image', or 'audio'
    search_query: str,     # Text string or URL
    video_index=None,
    audio_index=None,
    text_index=None,
    desc_index=None,
    embedding_model=None,
    caption_processor=None,
    caption_model=None,
    whisper_model=None,
    device: str = "cpu",
    top_k: int = 8,
    filter_dict: Optional[Dict] = None,
    merge_clips: bool = True,
    gap_seconds: int = 3
) -> List[Any]:
```

**Features:**
- Handles all three search types (text, image, audio)
- Downloads files from URLs automatically
- Routes to appropriate search functions:
  - `perform_text_search()` - queries all 4 indexes
  - `perform_image_search()` - queries video + description indexes
  - `perform_audio_search()` - queries audio + text indexes
- Merges overlapping clips automatically

### 2. `gpu/handler.py`
**Status:** ✅ Already Complete

**Key Functions:**

1. **Main Handler** - Routes requests based on `task_type`:
```python
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    input_data = job.get('input', {})
    task_type = input_data.get('task_type', 'embedding')
    
    if task_type == 'search':
        return handle_search(input_data)
    else:
        return handle_embedding(input_data)
```

2. **Search Handler** - Processes search requests:
```python
def handle_search(input_data: Dict[str, Any]) -> Dict[str, Any]:
    search_type = input_data.get('search_type')
    search_query = input_data.get('search_query')
    top_k = input_data.get('top_k', 8)
    filter_dict = input_data.get('filter')
    
    # Validates inputs
    # Initializes Pinecone indexes
    # Loads models (ImageBind, BLIP, Whisper if needed)
    # Performs search
    # Returns JSON response
```

## Files Updated in This Session

### 3. `gpu/handler.py` (Enhanced)
**Status:** ✅ Updated

**Changes:**
- Improved result serialization to handle both Pinecone Match objects and dict-like objects
- Better error handling for different result formats

**Code Snippet:**
```python
# Convert results to JSON-serializable format
serialized_results = []
for result in results:
    # Handle both Pinecone Match objects and dict-like objects
    if hasattr(result, 'score'):
        score = float(result.score)
    elif isinstance(result, dict):
        score = float(result.get('score', 0.0))
    else:
        score = 0.0
    
    if hasattr(result, 'metadata'):
        metadata = dict(result.metadata) if result.metadata else {}
    elif isinstance(result, dict):
        metadata = dict(result.get('metadata', {}))
    else:
        metadata = {}
    
    serialized_results.append({
        'score': score,
        'metadata': metadata
    })
```

### 4. `gpu/local.py` (Enhanced)
**Status:** ✅ Updated

**Changes:**
- Added support for testing search functionality
- Supports both embedding and search tasks
- Command-line and environment variable support

**Usage Examples:**

**Search via Environment Variables:**
```bash
export TASK_TYPE='search'
export SEARCH_TYPE='text'
export SEARCH_QUERY='a person running'
export TOP_K=8
python local.py
```

**Search via Command Line:**
```bash
# Text search
python local.py search text "a person running" 8

# Image search
python local.py search image "https://example.com/image.jpg" 8

# Audio search
python local.py search audio "https://example.com/audio.wav" 8
```

**Embedding Task (default):**
```bash
# Still works as before
python local.py <video_url> <transcript_url> [video_name]
```

## API Request Format

### Search Request
```json
{
  "input": {
    "task_type": "search",
    "search_type": "text" | "image" | "audio",
    "search_query": "text query or URL",
    "top_k": 8,
    "filter": {
      "video_name": "optional_filter"
    }
  }
}
```

### Embedding Request (unchanged)
```json
{
  "input": {
    "task_type": "embedding",
    "videoPathURL": "https://...",
    "transcriptPathURL": "https://...",
    "video_name": "video_name"
  }
}
```

## Response Format

### Search Response
```json
{
  "statusCode": 200,
  "body": {
    "status": "success",
    "search_type": "text",
    "results_count": 5,
    "results": [
      {
        "score": 0.95,
        "metadata": {
          "scene_uuid": "...",
          "video_doc_id": "...",
          "start_time": 10.5,
          "end_time": 15.2,
          ...
        }
      }
    ]
  }
}
```

## Search Types Details

### 1. Text Search (`search_type: "text"`)
- **Input:** Text string (e.g., "a person running")
- **Process:**
  - Generates text embedding using ImageBind
  - Queries all 4 indexes (video, audio, text, description)
  - Merges and scores results
- **Indexes Used:** All 4 indexes

### 2. Image Search (`search_type: "image"`)
- **Input:** Image URL or local file path
- **Process:**
  - Downloads image if URL provided
  - Generates image embedding using ImageBind
  - Generates caption using BLIP
  - Generates text embedding from caption
  - Queries video index (image embedding) + description index (caption embedding)
  - Merges and scores results
- **Indexes Used:** Video index, Description index

### 3. Audio Search (`search_type: "audio"`)
- **Input:** Audio URL or local file path
- **Process:**
  - Downloads audio if URL provided
  - Generates audio embedding using ImageBind
  - Transcribes audio using Whisper
  - Generates text embedding from transcript
  - Queries audio index (audio embedding) + text index (transcript embedding)
  - Merges and scores results using audio-specific scoring
- **Indexes Used:** Audio index, Text index

## Testing

### Test Search Locally
```bash
# Text search
python gpu/local.py search text "your search query"

# Image search
python gpu/local.py search image "https://example.com/image.jpg"

# Audio search
python gpu/local.py search audio "https://example.com/audio.wav"
```

### Test via Docker
```bash
# Build and run with test_local.sh (for embedding)
./test_local.sh

# For search, modify test_local.sh or use docker run directly
docker run --rm --gpus all \
  -v $(pwd)/.env:/app/.env \
  video-search-gpu:latest \
  python /local.py search text "your query"
```

## Summary

✅ **Complete Implementation:**
- `src/retrieval/search.py` - Core search logic
- `gpu/handler.py` - Request handling and routing
- `gpu/local.py` - Local testing support

✅ **All three search types supported:**
- Text search
- Image search  
- Audio search

✅ **Unified interface:**
- Single `search_videos()` function
- Single `handle_search()` handler
- Consistent request/response format


