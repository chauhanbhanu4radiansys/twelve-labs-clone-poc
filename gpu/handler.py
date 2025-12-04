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
    RunPod serverless handler for video search embeddings processing.
    
    Expected job format:
    {
        "input": {
            "videoPathURL": "https://...",  # Pre-signed S3 URL to video file
            "transcriptPathURL": "https://...",  # Pre-signed S3 URL to transcript JSON file
            "video_name": "video_name",  # Name of the video
            "video_id": "optional_video_id",  # Optional, will be auto-generated if None
            "id": "job_id",  # Job ID for notifications
            "tenant": "tenant_id",  # Tenant ID for notifications
            "snsTopicArn": "arn:aws:sns:...",  # SNS topic ARN for notifications
            "duration": 10.0,  # Optional: video duration in seconds
            "startTime": 0.0  # Optional: start time in seconds
        }
    }
    """
    try:
        # Extract input from job
        input_data = job.get('input', {})
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
                # Use local file path - download_from_url will detect it's a local file
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
            # For full video downloads, the process_video function handles it via URL
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
            
            result = {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success',
                    'message': 'Video embeddings processed successfully',
                    'video_name': video_name,
                    'video_id': video_id
                })
            }
            
            # Note: process_video handles cleanup of downloaded files internally
            # Only clean up if we downloaded a range and process_video didn't handle it
            # (process_video will clean up the video_path it gets from download_from_url)
            return result
            
        except Exception as e:
            error_message = str(e)
            print(f"❌ Error processing embeddings: {error_message}")
            import traceback
            try:
                traceback.print_exc()
            except (OSError, ValueError) as exc:
                # stderr might be closed, try to print to stdout instead
                try:
                    print(f"Traceback (stderr unavailable): {exc}", file=sys.stdout)
                except:
                    pass  # Both stdout and stderr unavailable, ignore
            
            # Send failure notification
            if tenant and job_id and sns_topic_arn:
                notify(tenant, job_id, False, sns_topic_arn, {'error': error_message})
            
            # Cleanup downloaded range file if process_video failed before handling it
            if downloaded_video_path and os.path.exists(downloaded_video_path):
                try:
                    cleanup_temp_files(downloaded_video_path)
                except Exception as cleanup_error:
                    print(f"⚠ Warning: Could not clean up downloaded file: {cleanup_error}")
            
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'status': 'error',
                    'error': error_message
                })
            }
        
    except Exception as e:
        error_message = str(e)
        print(f"❌ Error in handler: {error_message}")
        import traceback
        try:
            traceback.print_exc()
        except (OSError, ValueError) as exc:
            # stderr might be closed, try to print to stdout instead
            try:
                print(f"Traceback (stderr unavailable): {exc}", file=sys.stdout)
            except:
                pass  # Both stdout and stderr unavailable, ignore
        
        # Send failure notification if we have the required info
        input_data = job.get('input', {}) if isinstance(job, dict) else {}
        tenant = input_data.get('tenant')
        job_id = input_data.get('id')
        sns_topic_arn = input_data.get('snsTopicArn')
        
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, False, sns_topic_arn, {'error': error_message})
        
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
