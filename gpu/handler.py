import os
import sys
import json
import subprocess
import requests
import shutil
import time
from typing import Dict, Any

# Optional runpod import (only needed for RunPod serverless deployment)
try:
    import runpod
except ImportError:
    runpod = None

# Add parent directory to Python path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notify import notify
from src.embedding import process_video
from dotenv import load_dotenv
import boto3

load_dotenv()

def upload(inpath, outpath, type):
    try:
        with open(inpath, 'rb') as f:
            headers = {'Content-Type': 'video/mp4'} if type == 'video' else {}
            r = requests.put(outpath, data=f, headers=headers, timeout=120)
            r.raise_for_status()
            print(f"✓ Uploaded {type} {inpath} → {outpath}")
            return True
    except Exception as e:
        print(f"✗ Upload failed for {type}: {e}")
        return False

def download_with_range(inpath, outpath, startTime, duration):
    """
    Download a specific range of video from S3 with ffmpeg streaming.
    Falls back to full download + local trim if streaming fails.
    """
    try:
        return _download_with_range_streaming(inpath, outpath, startTime, duration)
    except Exception as e:
        print(f"Streaming method failed: {e}")
        print("Attempting fallback method...")
        return _download_with_range_fallback(inpath, outpath, startTime, duration)

def _download_with_range_streaming(inpath, outpath, startTime, duration, max_retries=2):
    """Download with range using ffmpeg streaming from presigned URL"""
    
    for attempt in range(max_retries + 1):
        try:
            # Improved ffmpeg command with better stability parameters
            # Force MP4 encoding to ensure compatibility regardless of input format
            cmd = [
                'ffmpeg',
                '-y',                                    # Overwrite output file
                '-protocol_whitelist', 'pipe,file,http,https,tcp,tls,crypto',  # Allow HTTPS protocols
                '-reconnect', '1',                       # Enable reconnection on network issues
                '-reconnect_at_eof', '1',               # Reconnect at end of file
                '-reconnect_streamed', '1',             # Reconnect for streamed content
                '-reconnect_delay_max', '5',            # Max delay between reconnects
                '-ss', str(startTime),                  # Start time
                '-i', inpath,                    # Input URL
                '-t', str(duration),                    # Duration
                '-c:v', 'libx264',                      # Use H.264 video codec for MP4 compatibility
                '-c:a', 'aac',                          # Use AAC audio codec for MP4 compatibility
                '-preset', 'fast',                      # Fast encoding preset
                '-crf', '23',                           # Constant rate factor for good quality
                '-avoid_negative_ts', 'make_zero',      # Handle negative timestamps
                '-fflags', '+genpts',                   # Generate presentation timestamps
                '-movflags', '+faststart',              # Optimize for streaming
                outpath
            ]
            
            
            # Run with timeout and better error handling
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"✓ Successfully downloaded range using streaming method")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"Attempt {attempt + 1} timed out")
            if attempt == max_retries:
                raise Exception("ffmpeg streaming timed out after all retries")
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed with return code {e.returncode}")
            print(f"stderr: {e.stderr}")
            if attempt == max_retries:
                raise Exception(f"ffmpeg streaming failed: {e.stderr}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries:
                raise e
                
        # Wait before retry
        if attempt < max_retries:
            time.sleep(2 ** attempt)  # Exponential backoff

def _download_with_range_fallback(inpath, outpath, startTime, duration):
    """
    Fallback method: download full file first, then trim locally.
    More reliable but uses more bandwidth and storage.
    """
    import tempfile
    
    print(f"Using fallback method: full download + local trim")
    
    # Create temporary file for full download
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_full_path = temp_file.name
    
    try:
        # Download full file
        print(f"Downloading full file to temporary location...")
        download(inpath, temp_full_path)
        
        # Trim locally using ffmpeg and encode as MP4
        print(f"Trimming video locally: {startTime}s-{float(startTime) + float(duration)}s")
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(startTime),
            '-i', temp_full_path,
            '-t', str(duration),
            '-c:v', 'libx264',                      # Use H.264 video codec for MP4 compatibility
            '-c:a', 'aac',                          # Use AAC audio codec for MP4 compatibility
            '-preset', 'fast',                      # Fast encoding preset
            '-crf', '23',                           # Constant rate factor for good quality
            '-avoid_negative_ts', 'make_zero',
            '-movflags', '+faststart',              # Optimize for streaming
            outpath
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout for local operation
        )
        
        print(f"✓ Successfully created trimmed video using fallback method")
        return True
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_full_path):
                os.unlink(temp_full_path)
                print(f"✓ Cleaned up temporary file: {temp_full_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not clean up temporary file {temp_full_path}: {e}")


# Check FFmpeg availability at startup
def check_ffmpeg_availability():
    """Check if FFmpeg is available and log the result"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ FFmpeg is available: {result.stdout.split()[0:3]}")
            return True
        else:
            print(f"✗ FFmpeg check failed with return code {result.returncode}")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        # Try to find FFmpeg in common locations
        common_paths = ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/bin/ffmpeg']
        for path in common_paths:
            if os.path.exists(path):
                print(f"✓ Found FFmpeg at: {path}")
                return True
        print("✗ FFmpeg not found in any common location")
        return False
    except Exception as e:
        print(f"✗ Error checking FFmpeg: {e}")
        return False

# Check FFmpeg at module load time
check_ffmpeg_availability()

def download(inpath, outpath):
    r = requests.get(inpath, stream=True, timeout=300)
    r.raise_for_status()
    with open(outpath, 'wb') as f:
        f.write(r.content)
    return True

def cleanup_temp_files(*file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✓ Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not clean up {file_path}: {e}")

def get_video_resolution(video_path):
    """
    Get video resolution using ffprobe.
    
    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        width, height = map(int, result.stdout.strip().split(','))
        print(f"Video resolution: {width}x{height}")
        return width, height
    except Exception as e:
        print(f"⚠ Warning: Could not get video resolution: {e}")
        return None

def is_2k_or_higher(width, height):
    """
    Check if video is 2K or higher resolution.
    2K typically means width >= 2048 or height >= 1440 (QHD).
    """
    if width is None or height is None:
        return False
    # 2K: width >= 2048 (DCI 2K) or height >= 1440 (QHD/1440p)
    return width >= 2048 or height >= 1440

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for video search operations.
    
    Supports the following task types:
    - 'embedding' (default): Process video and generate embeddings
    - 'search': Search across indexed videos
    - 'analyse' or 'analyze': Analyze video content using LLM
    
    Expected job format for 'embedding':
    {
        "input": {
            "task_type": "embedding",  # Optional, defaults to 'embedding'
            "videoPathURL": "https://...",
            "transcriptPathURL": "https://...",
            "video_name": "video_name",
            "video_id": "optional_video_id",
            "id": "job_id",
            "tenant": "tenant_id",
            "snsTopicArn": "arn:aws:sns:...",
            "duration": 10.0,  # Optional
            "startTime": 0.0  # Optional
        }
    }
    
    Expected job format for 'search':
    {
        "input": {
            "task_type": "search",
            "search_type": "text" | "image" | "audio",
            "query": "text query or URL",
            "top_k": 8,  # Optional, default 8
            "filter": {"video_name": "..."}  # Optional metadata filter
        }
    }
    
    Expected job format for 'analyse':
    {
        "input": {
            "task_type": "analyse",
            "attachment_id": "video_doc_id",
            "query": "your analysis prompt"
        }
    }
    """
    try:
        input_data = job.get('input', {})
        task_type = input_data.get('task_type', 'embedding')
        
        if task_type == 'search':
            return handle_search(input_data)
        elif task_type == 'analyse' or task_type == 'analyze':
            return handle_analyse(input_data)
        else:
            # Default to embedding task
            return handle_embedding(input_data)
            
    except Exception as e:
        error_message = str(e)
        print(f"❌ Error in handler: {error_message}")
        import traceback
        try:
            traceback.print_exc()
        except (OSError, ValueError):
            pass
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_message
            })
        }


def handle_search(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle search requests.
    
    Supports both production (S3 pre-signed URLs) and local testing (file paths):
    - For 'image' search: query can be S3 pre-signed URL (https://...) or local file path
    - For 'audio' search: query can be S3 pre-signed URL (https://...) or local file path
    - For 'text' search: query is a text string
    
    Args:
        input_data: Dictionary with search_type, query, and optional parameters
            - search_type: 'text', 'image', or 'audio'
            - query: For image/audio: S3 pre-signed URL or local file path
                    For text: search query string
            - top_k: Optional, number of results (default: 8)
            - filter: Optional metadata filter dictionary
        
    Returns:
        JSON response with search results
    """
    from src.retrieval.search import search_videos
    from src.embedding.models import load_imagebind_model, load_captioning_model, load_whisper_model
    from src.embedding.pipeline import init_pinecone_indexes
    
    search_type = input_data.get('search_type')
    query = input_data.get('query') or input_data.get('search_query')  # Support both for backward compatibility
    top_k = input_data.get('top_k', 8)
    filter_dict = input_data.get('filter')
    
    # Validate required parameters
    if not search_type:
        raise ValueError("Missing required parameter: search_type")
    if not query:
        raise ValueError("Missing required parameter: query")
    if search_type not in ('text', 'image', 'audio'):
        raise ValueError(f"Invalid search_type: {search_type}. Must be 'text', 'image', or 'audio'.")
    
    print("=" * 80)
    print(f"Starting Search: type={search_type}")
    # For URLs, show truncated version; for local paths, show full path
    if query.startswith('http://') or query.startswith('https://'):
        print(f"Query URL: {query[:100]}{'...' if len(query) > 100 else ''}")
    else:
        print(f"Query: {query}")
    print("=" * 80)
    
    # Initialize Pinecone indexes
    video_index, audio_index, text_index, desc_index = init_pinecone_indexes()
    if not all([video_index, audio_index, text_index, desc_index]):
        raise Exception("Failed to initialize Pinecone indexes")
    
    # Load models
    embedding_model, device = load_imagebind_model()
    caption_processor, caption_model = load_captioning_model()
    whisper_model = load_whisper_model() if search_type == 'audio' else None
    
    # Perform search with scene merging enabled (matching UI behavior)
    # Scene merging combines overlapping or nearby clips (within 3 seconds) from the same video
    results = search_videos(
        search_type=search_type,
        query=query,  # Changed from search_query to query
        video_index=video_index,
        audio_index=audio_index,
        text_index=text_index,
        desc_index=desc_index,
        embedding_model=embedding_model,
        caption_processor=caption_processor,
        caption_model=caption_model,
        whisper_model=whisper_model,
        device=device,
        top_k=top_k,
        filter_dict=filter_dict,
        merge_clips=True,  # Enable scene merging (same as UI) - merges overlapping/nearby clips
        gap_seconds=3  # Merge clips within 3 seconds of each other
    )
    
    # Convert results to JSON-serializable format matching app-code-ref.py UI format
    # Results are already sorted by score (highest first) and merged from search_videos()
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
        
        # Categorize score (matching app-code-ref.py logic)
        if score >= 0.5:
            category = "HIGH"
        elif score >= 0.3:
            category = "MEDIUM"
        else:
            category = "LOW"
        
        # Format timestamps as HH:MM:SS (matching UI format)
        start_time = metadata.get('start_time', 0.0)
        end_time = metadata.get('end_time', 0.0)
        start_timestamp = time.strftime('%H:%M:%S', time.gmtime(start_time)) if start_time else "00:00:00"
        end_timestamp = time.strftime('%H:%M:%S', time.gmtime(end_time)) if end_time else "00:00:00"
        
        # Build result object matching UI display format
        result_obj = {
            'score': round(score, 3),  # Round to 3 decimal places
            'category': category,  # HIGH, MEDIUM, or LOW
            'video_name': metadata.get('video_name', 'Unknown'),
            'video_id': metadata.get('video_id', metadata.get('video_doc_id', '')),
            'start_time': start_time,  # Raw seconds (float)
            'end_time': end_time,  # Raw seconds (float)
            'start_timestamp': start_timestamp,  # Formatted as HH:MM:SS
            'end_timestamp': end_timestamp,  # Formatted as HH:MM:SS
            'time_range': f"{start_timestamp} - {end_timestamp}",  # Display format
            'transcript': metadata.get('transcript', ''),
            'description': metadata.get('description', ''),
            'source': metadata.get('source', 'N/A'),  # Which modalities matched (e.g., "video, text")
            'scene_uuid': metadata.get('scene_uuid', ''),
            'scene_index': metadata.get('scene_index', -1),
            # Include full metadata for backward compatibility
            'metadata': metadata
        }
        
        serialized_results.append(result_obj)
    
    # Ensure results are sorted by score (highest first) - should already be sorted, but double-check
    serialized_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Search complete: {len(serialized_results)} results found (with scene merging enabled)")
    print("=" * 80)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'success',
            'search_type': search_type,
            'results_count': len(serialized_results),
            'scene_merging': True,  # Indicates that overlapping/nearby scenes were merged
            'merge_gap_seconds': 3,  # Clips within 3 seconds were merged
            'results': serialized_results
        }, indent=2)  # Pretty print JSON for readability
    }


def handle_analyse(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle video analysis requests using LLM.
    
    Args:
        input_data: Dictionary with attachment_id and query
        
    Returns:
        JSON response with analysis results
    """
    from src.retrieval.analyze import analyze_video, get_video_data_from_pinecone
    from src.retrieval.search import search_videos
    from src.embedding.models import load_imagebind_model, load_captioning_model
    from src.embedding.pipeline import init_pinecone_indexes
    
    attachment_id = input_data.get('attachment_id')
    query = input_data.get('query')
    
    # Validate required parameters
    if not attachment_id:
        raise ValueError("Missing required parameter: attachment_id")
    if not query:
        raise ValueError("Missing required parameter: query")
    
    print("=" * 80)
    print("Starting Video Analysis")
    print("=" * 80)
    print(f"Attachment ID: {attachment_id}")
    print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    print("=" * 80)
    
    # Initialize Pinecone indexes
    print("Initializing Pinecone indexes...")
    video_index, audio_index, text_index, desc_index = init_pinecone_indexes()
    
    if not all([video_index, audio_index, text_index, desc_index]):
        raise Exception("Failed to initialize Pinecone indexes. Check credentials.")
    
    # Fetch video data from Pinecone (transcript segments and duration)
    print(f"Fetching video data from Pinecone for attachment_id: {attachment_id}...")
    transcript_segments, video_duration = get_video_data_from_pinecone(
        video_doc_id=attachment_id,
        text_index=text_index,
        max_scenes=1000
    )
    
    if not transcript_segments:
        raise ValueError(f"No transcript data found for attachment_id: {attachment_id}")
    
    if not video_duration:
        raise ValueError(f"Could not determine video duration for attachment_id: {attachment_id}")
    
    print(f"Found {len(transcript_segments)} transcript segments")
    print(f"Video duration: {video_duration:.2f} seconds")
    
    # Initialize OpenAI client
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for analysis")
    
    try:
        import openai
        openai_client = openai.OpenAI(api_key=openai_api_key)
    except ImportError:
        raise ImportError("OpenAI library is not installed. Install with: pip install openai")
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    # Load models for potential RAG-based search (if needed for SPECIFIC_QUESTION)
    print("Loading models for analysis...")
    embedding_model, device = load_imagebind_model()
    caption_processor, caption_model = load_captioning_model()
    
    if not embedding_model:
        raise Exception("Failed to load embedding model")
    
    print(f"Using device: {device}")
    
    # Perform analysis
    # The analyze_video function will:
    # 1. Detect query intent (SPECIFIC_QUESTION vs HOLISTIC_REQUEST)
    # 2. For SPECIFIC_QUESTION: Perform RAG-based search and use results
    # 3. For HOLISTIC_REQUEST: Use full transcript segments
    print("Performing analysis...")
    
    # For SPECIFIC_QUESTION, we may need to perform a search first
    # Let analyze_video handle this internally by passing the necessary components
    analysis_result = analyze_video(
        prompt=query,
        video_doc_id=attachment_id,
        transcript_segments=transcript_segments,
        video_duration=video_duration,
        search_results=None,  # Will be generated internally if needed
        embedding_model=embedding_model,
        device=device,
        openai_client=openai_client,
        text_index=text_index,
        desc_index=desc_index
    )
    
    if not analysis_result:
        raise Exception("Analysis failed - no result returned")
    
    print("Analysis completed successfully")
    
    # Format response
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'success',
            'attachment_id': attachment_id,
            'query': query,
            'video_duration': video_duration,
            'transcript_segments_count': len(transcript_segments),
            'analysis': analysis_result
        }, indent=2)
    }


def handle_embedding(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle embedding/video processing requests.
    
    Args:
        input_data: Dictionary with video processing parameters
        
    Returns:
        JSON response with processing status
    """
    videoPathURL = input_data.get('videoPathURL')
    transcriptPathURL = input_data.get('transcriptPathURL')
    video_name = input_data.get('video_name')
    video_id = input_data.get('video_id')
    job_id = input_data.get('id')
    tenant = input_data.get('tenant')
    sns_topic_arn = input_data.get('snsTopicArn')
    duration = input_data.get('duration', None)
    startTime = input_data.get('startTime', None)
    
    # Validate required parameters
    if not videoPathURL:
        raise ValueError("Missing required parameter: videoPathURL")
    if not transcriptPathURL:
        raise ValueError("Missing required parameter: transcriptPathURL")
    if not video_name:
        raise ValueError("Missing required parameter: video_name")
    
    print("=" * 80)
    print("Starting Video Search Embeddings Processing")
    print("=" * 80)
    print(f"Video URL: {videoPathURL[:100]}...")
    print(f"Transcript URL: {transcriptPathURL[:100]}...")
    print(f"Video Name: {video_name}")
    print(f"Video ID: {video_id or 'auto-generated'}")
    print(f"Job ID: {job_id}")
    print("=" * 80)
    
    # Set up temporary file paths in /tmp
    local_video_path = '/tmp/input_video.mp4'
    downloaded_video_path = None
    
    # Download video if range is specified (for partial video processing)
    if duration is not None and startTime is not None:
        try:
            print(f"Downloading video range: {startTime}s to {startTime + duration}s")
            download_with_range(videoPathURL, local_video_path, startTime, duration)
            downloaded_video_path = local_video_path
            videoPathURL = local_video_path
        except Exception as e:
            print(f"✗ Failed to download video range: {e}")
            if tenant and job_id and sns_topic_arn:
                notify(tenant, job_id, False, sns_topic_arn, {'error': f'Download failed: {str(e)}'})
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'status': 'error',
                    'error': f'Download failed: {str(e)}'
                })
            }
    else:
        print("Using full video URL for processing")
    
    # Status callback function for progress updates
    def status_callback(message: str):
        print(f"[STATUS] {message}")
    
    # Run the embedding pipeline
    try:
        process_video(
            videoPathURL=videoPathURL,
            transcriptPathURL=transcriptPathURL,
            video_name=video_name,
            video_id=video_id,
            status_callback=status_callback
        )
        
        print("=" * 80)
        print("Video Search Embeddings Processing Completed Successfully")
        print("=" * 80)
        
        # Send success notification
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, True, sns_topic_arn, {
                'video_name': video_name,
                'video_id': video_id
            })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'message': 'Video embeddings processed successfully',
                'video_name': video_name,
                'video_id': video_id
            })
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"❌ Error processing embeddings: {error_message}")
        import traceback
        try:
            traceback.print_exc()
        except (OSError, ValueError):
            pass
        
        # Send failure notification
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, False, sns_topic_arn, {'error': error_message})
        
        # Cleanup downloaded range file if process_video failed before handling it
        if downloaded_video_path and os.path.exists(downloaded_video_path):
            try:
                cleanup_temp_files(downloaded_video_path)
            except Exception:
                pass
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_message
            })
        }


# For RunPod serverless (only start if runpod is available)
if runpod is not None:
    runpod.serverless.start({"handler": handler})

