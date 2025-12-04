import os
import sys
from dotenv import load_dotenv

# Set runtime environment
os.environ['RUNTIME_ENVIRONMENT'] = 'local'

# Get project root directory (parent of gpu directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from project root
# Try project root first, then current directory as fallback
env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()  # Fallback to current directory

# Add parent directory to Python path
sys.path.insert(0, PROJECT_ROOT)

# GPU Verification at startup
try:
    import torch
    print("\n" + "="*60)
    print("GPU VERIFICATION")
    print("="*60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ GPU Device: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✅ PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CRITICAL: CUDA NOT AVAILABLE - Running on CPU!")
        print("   This will be 10-100x slower than GPU")
        print("   Expected processing time: 10-30+ minutes")
        print("")
        print("   To enable GPU:")
        print("   1. Check Docker GPU access:")
        print("      docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi")
        print("   2. Install NVIDIA Container Toolkit if needed")
        print("   3. Restart Docker: sudo systemctl restart docker")
    print("="*60 + "\n")
except ImportError:
    print("⚠️  Warning: PyTorch not available, cannot verify GPU")
except Exception as e:
    print(f"⚠️  Warning: Could not verify GPU: {e}")

from handler import handler

if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO SEARCH GPU CONTAINER - LOCAL TESTING")
    print("=" * 60)
    print()
    
    # Get URLs from environment variables or use placeholders
    video_url = os.getenv('VIDEO_URL', '')
    transcript_url = os.getenv('TRANSCRIPT_URL', '')
    video_name = os.getenv('VIDEO_NAME', 'test_video')
    video_id = os.getenv('VIDEO_ID', None)
    job_id = os.getenv('JOB_ID', 'test-job-123')
    tenant = os.getenv('TENANT', 'test-tenant')
    sns_topic_arn = os.getenv('SNS_TOPIC_ARN', '')
    
    # If URLs not in environment, check command line arguments
    if not video_url and len(sys.argv) > 1:
        video_url = sys.argv[1]
    if not transcript_url and len(sys.argv) > 2:
        transcript_url = sys.argv[2]
    if len(sys.argv) > 3:
        video_name = sys.argv[3]
    
    # If still no URLs, use placeholders with instructions
    if not video_url or not transcript_url:
        print("⚠️  No URLs provided!")
        print()
        print("Usage options:")
        print("  1. Set environment variables:")
        print("     export VIDEO_URL='https://...'")
        print("     export TRANSCRIPT_URL='https://...'")
        print("     export VIDEO_NAME='my_video'")
        print("     python local.py")
        print()
        print("  2. Pass as command line arguments:")
        print("     python local.py <video_url> <transcript_url> [video_name]")
        print()
        print("  3. Update the placeholders below and run:")
        print()
        
        # ========== PLACEHOLDER: Update these with your actual pre-signed S3 URLs ==========
        video_url = video_url or "PLACEHOLDER_PRE_SIGNED_VIDEO_URL"
        transcript_url = transcript_url or "PLACEHOLDER_PRE_SIGNED_TRANSCRIPT_URL"
        video_name = video_name or "PLACEHOLDER_VIDEO_NAME"
    
    # Create job payload
    video_search_job = {
        "input": {
            "videoPathURL": video_url,
            "transcriptPathURL": transcript_url,
            "video_name": video_name,
            "video_id": video_id,
            "id": job_id,
            "tenant": tenant,
            "snsTopicArn": sns_topic_arn if sns_topic_arn else None
        }
    }
    
    print("Job Configuration:")
    print(f"  Video URL: {video_url[:80]}..." if len(video_url) > 80 else f"  Video URL: {video_url}")
    print(f"  Transcript URL: {transcript_url[:80]}..." if len(transcript_url) > 80 else f"  Transcript URL: {transcript_url}")
    print(f"  Video Name: {video_name}")
    print(f"  Video ID: {video_id or 'auto-generated'}")
    print(f"  Job ID: {job_id}")
    print(f"  Tenant: {tenant}")
    if sns_topic_arn:
        print(f"  SNS Topic: {sns_topic_arn}")
    print()
    print("=" * 60)
    print("Starting Processing...")
    print("=" * 60)
    print()
    
    try:
        result = handler(video_search_job)
        print()
        print("=" * 60)
        print("Result:")
        print("=" * 60)
        print(result)
    except Exception as e:
        print()
        print("=" * 60)
        print("Error occurred:")
        print("=" * 60)
        print(f"❌ {e}")
        import traceback
        try:
            traceback.print_exc()
        except (OSError, ValueError) as exc:
            # stderr might be closed, try to print to stdout instead
            try:
                print(f"Traceback (stderr unavailable): {exc}", file=sys.stdout)
            except:
                pass  # Both stdout and stderr unavailable, ignore
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("USAGE")
    print("=" * 60)
    print()
    print("The handler() function processes video embeddings for search:")
    print()
    print("  Required fields in input:")
    print("    - videoPathURL: Pre-signed S3 URL to video file")
    print("    - transcriptPathURL: Pre-signed S3 URL to transcript JSON file")
    print("    - video_name: Name of the video")
    print()
    print("  Optional fields in input:")
    print("    - video_id: Unique identifier (auto-generated if None)")
    print("    - id: Job ID for notifications")
    print("    - tenant: Tenant ID for notifications")
    print("    - snsTopicArn: SNS topic ARN for notifications")
    print("    - duration: Video duration in seconds (for range downloads)")
    print("    - startTime: Start time in seconds (for range downloads)")
    print()
    print("=" * 60)
