# Unified Video Handler Implementation Plan

## Goal
Extend `gpu/handler.py` to support three distinct operations:
1.  **Embedding Creation** (Existing functionality)
2.  **Search** (New: Text, Image, Audio search across Pinecone indexes)
3.  **Analysis** (New: LLM-based analysis of specific videos)

## User Review Required
> [!IMPORTANT]
> This change requires adding `openai` dependency to the Docker image if it's not already there.
> The input payload format will change to include a `task_type` field.

## Proposed Changes

### `gpu/handler.py`

#### [MODIFY] `gpu/handler.py`
-   **Imports**: Import necessary functions from `src.retrieval`:
    -   `from src.retrieval.search import search_videos`
    -   `from src.retrieval.analyze import analyze_video`
    -   Ensure `openai` and `PIL.Image` are imported.
-   **Handler Logic**:
    -   Update `handler(job)` to check for `task_type` in `job['input']`.
    -   **Case `embedding`** (Default): Call existing `process_video` logic.
    -   **Case `search`**:
        -   Extract parameters: `search_type` (text, image, audio) and `search_query` (text or URL).
        -   Call `search_videos` with `search_type` and `search_query`.
        -   `search_videos` will handle downloading if `search_query` is a URL for image/audio types.
        -   Return results.
    -   **Case `analysis`**:
        -   Extract parameters: `video_name`, `prompt`, `transcript_url` (optional), `video_duration` (optional).
        -   If `transcript_url` is provided, download transcript segments.
        -   If not, attempt to retrieve context from Pinecone using `video_name` (less reliable for full transcript).
        -   Call `analyze_video`.
        -   Return analysis JSON.

### `gpu/requirements.txt` (if exists) or `gpu/Dockerfile`
-   Ensure `openai` is installed.

## Verification Plan

### Automated Tests
-   Create `test_unified_handler.py`:
    -   **Test Embedding**: Mock `process_video` and verify it's called when `task_type="embedding"`.
    -   **Test Search**: Mock Pinecone and Embedding model, send `task_type="search"` with text query, verify results structure.
    -   **Test Analysis**: Mock OpenAI and MongoDB/S3 (for transcript), send `task_type="analysis"`, verify JSON response.

### Manual Verification
-   Run `test_local.sh` (modified to support new inputs) to verify the container builds and runs.
