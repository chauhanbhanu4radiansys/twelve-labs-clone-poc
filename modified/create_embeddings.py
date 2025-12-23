#!/usr/bin/env python3
"""
Create Embeddings Script
Generates embeddings for videos using ImageBind and uploads to Pinecone.

Usage:
    # Using local files:
    python create_embeddings.py --video /path/to/video.mp4 --transcript /path/to/transcript.json --name "my_video"
    
    # Using URLs:
    python create_embeddings.py --video-url "https://..." --transcript-url "https://..." --name "my_video"
    
    # With video ID:
    python create_embeddings.py --video /path/to/video.mp4 --transcript /path/to/transcript.json --name "my_video" --video-id "custom-id-123"
    
Environment Variables (from .env):
    PINECONE_API_KEY - Required
    PINECONE_FRAME_INDEX - Video index name (default: video-search)
    PINECONE_AUDIO_INDEX - Audio index name (default: audio-search)
    PINECONE_TRANSCRIPT_INDEX - Text index name (default: text-search)
    PINECONE_DESCRIPTION_INDEX - Description index name (default: desc-search)
"""

import os
import sys
import argparse
import uuid
from pathlib import Path
import torch

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv not available, manually parse .env
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import embedding pipeline
from src.embedding import process_video
from src.embedding.config import PINECONE_CONFIG


def create_presigned_url_for_local_file(file_path: str) -> str:
    """
    For local files, we'll use a file:// URL or handle it directly.
    Since process_video expects URLs, we'll need to handle local files differently.
    Actually, looking at the code, process_video downloads from URLs.
    For local files, we can create a simple file handler or modify the approach.
    """
    # Return as-is for now - we'll handle local files by creating temp URLs or modifying download logic
    return file_path


def main():
    parser = argparse.ArgumentParser(
        description='Create embeddings for videos and upload to Pinecone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using local files:
  python create_embeddings.py --video video.mp4 --transcript transcript.json --name "my_video"
  
  # Using URLs:
  python create_embeddings.py --video-url "https://s3.amazonaws.com/bucket/video.mp4?..." --transcript-url "https://..." --name "my_video"
  
  # Mixed: local video + S3 transcript URL:
  python create_embeddings.py --video video.mp4 --transcript-url "https://s3.amazonaws.com/bucket/transcript.json?..." --name "my_video"
  
  # Mixed: S3 video URL + local transcript:
  python create_embeddings.py --video-url "https://s3.amazonaws.com/bucket/video.mp4?..." --transcript transcript.json --name "my_video"
  
  # With custom video ID:
  python create_embeddings.py --video video.mp4 --transcript transcript.json --name "my_video" --video-id "custom-123"
        """
    )
    
    # Input options - can mix local files and URLs
    parser.add_argument('--video', type=str, default=None,
                       help='Path to local video file')
    parser.add_argument('--video-url', type=str, default=None,
                       help='URL to video file (S3 pre-signed URL or HTTP/HTTPS)')
    
    parser.add_argument('--transcript', type=str, default=None,
                       help='Path to local transcript JSON file')
    parser.add_argument('--transcript-url', type=str, default=None,
                       help='URL to transcript JSON file (S3 pre-signed URL or HTTP/HTTPS)')
    
    parser.add_argument('--name', '--video-name', dest='video_name', type=str, required=True,
                       help='Name/identifier for the video')
    parser.add_argument('--video-id', type=str, default=None,
                       help='Unique video ID (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Validate that at least one video source is provided
    if not args.video and not args.video_url:
        print("Error: Either --video or --video-url must be provided")
        return 1
    
    # Validate that at least one transcript source is provided
    if not args.transcript and not args.transcript_url:
        print("Error: Either --transcript or --transcript-url must be provided")
        return 1
    
    # Determine video and transcript sources
    video_source = args.video or args.video_url
    transcript_source = args.transcript or args.transcript_url
    
    # Check if using local files (for either video or transcript)
    using_local_files = args.video is not None or args.transcript is not None
    
    # Validate local files exist
    if args.video and not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if args.transcript and not os.path.exists(args.transcript):
        print(f"Error: Transcript file not found: {args.transcript}")
        return 1
    
    # Check environment variables
    if not PINECONE_CONFIG['api_key']:
        print("Error: PINECONE_API_KEY not found in .env file")
        return 1
    
    # Generate video ID if not provided
    video_id = args.video_id or str(uuid.uuid4())
    
    print("=" * 80)
    print("CREATE EMBEDDINGS")
    print("=" * 80)
    print(f"Video Name: {args.video_name}")
    print(f"Video ID: {video_id}")
    if args.video:
        print(f"Video File: {args.video}")
    else:
        print(f"Video URL: {args.video_url[:100]}...")
    if args.transcript:
        print(f"Transcript File: {args.transcript}")
    else:
        print(f"Transcript URL: {args.transcript_url[:100]}...")
    print("=" * 80)
    print()
    
    # Handle local files - need to create temporary URLs or modify download logic
    # For now, we'll use the existing process_video which expects URLs
    # If local files are provided, we'll need to serve them or convert to URLs
    
    if using_local_files:
        print("⚠️  Local files detected. Creating temporary file URLs...")
        print("   Note: For production, use S3 pre-signed URLs instead.")
        print()
        
        # For local files, we need to modify the approach
        # Option 1: Use file:// URLs (won't work with download_from_url)
        # Option 2: Copy files to a temp location and serve them
        # Option 3: Modify download logic to handle local paths
        
        # Let's check if the download function can handle local paths
        # Looking at download.py, it uses requests.get() which won't work with file://
        # So we'll need to handle this differently
        
        # For now, let's create a wrapper that handles local files
        from src.embedding.download import download_from_url, download_transcript_from_url
        import shutil
        import tempfile
        
        # Handle mixed inputs: local video + URL transcript, or both local, etc.
        video_path = args.video if args.video else None
        
        # If video is URL, download it first
        if args.video_url:
            print("Downloading video from URL...")
            from src.embedding.download import download_from_url
            video_path = download_from_url(args.video_url, progress_callback=print)
            if not video_path or not os.path.exists(video_path):
                print(f"Error: Failed to download video from URL")
                return 1
            print(f"✓ Video downloaded to: {video_path}")
        
        # Process with local video file (either provided directly or downloaded)
        print("Processing video...")
        
        # Import the necessary components
        from src.embedding.pipeline import (
            init_pinecone_indexes,
            process_video_chunk,
            split_video_into_chunks
        )
        from src.embedding.models import load_imagebind_model, load_captioning_model
        from src.embedding.processing import (
            detect_scenes_with_adaptive_detector,
            extract_audio_from_video,
            prepare_scene_data
        )
        from src.embedding.upload import PineconeUploadManager
        from src.embedding.pipeline import process_batch
        from pydub import AudioSegment
        import json
        import concurrent.futures
        
        try:
            # Initialize Pinecone
            print("Initializing Pinecone indexes...")
            video_index, audio_index, text_index, desc_index = init_pinecone_indexes()
            
            if not all([video_index, audio_index, text_index, desc_index]):
                print("Error: Failed to initialize Pinecone indexes")
                return 1
            
            print("✓ Indexes initialized")
            
            # Load transcript - handle both local file and URL
            transcript_segments = None
            if args.transcript:
                # Local transcript file
                transcript_path = args.transcript
                print(f"Loading transcript from local file: {transcript_path}")
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
                    # Handle different transcript formats
                    if isinstance(transcript_data, list):
                        transcript_segments = transcript_data
                    elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
                        transcript_segments = transcript_data['segments']
                    else:
                        print("Warning: Unknown transcript format")
                print(f"✓ Loaded {len(transcript_segments) if transcript_segments else 0} transcript segments")
            elif args.transcript_url:
                # Download transcript from URL
                print(f"Downloading transcript from URL...")
                from src.embedding.download import download_transcript_from_url
                transcript_segments = download_transcript_from_url(args.transcript_url)
                if transcript_segments:
                    print(f"✓ Loaded {len(transcript_segments)} transcript segments from URL")
                else:
                    print("Warning: Could not download transcript from URL")
            
            # Load models
            print("Loading models...")
            
            # Check CUDA availability first
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Count: {torch.cuda.device_count()}")
                try:
                    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
                    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
                except Exception as e:
                    print(f"Warning: Could not get GPU details: {e}")
            print()
            
            embedding_model, device = load_imagebind_model()
            # Load caption model on the same device as embedding model
            caption_processor, caption_model = load_captioning_model(device=device)
            
            if not embedding_model:
                print("Error: Failed to load ImageBind model")
                return 1
            
            # Verify models are actually on the correct device and fix if needed
            print("\n" + "="*80)
            print("VERIFYING GPU USAGE...")
            
            # Check actual device of embedding model
            try:
                embedding_device = next(embedding_model.parameters()).device
                actual_device_type = embedding_device.type
                actual_device_index = embedding_device.index if embedding_device.index is not None else 0
                
                # If CUDA is available but model is on CPU, try to move it
                if torch.cuda.is_available() and actual_device_type == 'cpu':
                    print("⚠️  WARNING: CUDA available but model is on CPU!")
                    print("  Attempting to move models to GPU...")
                    try:
                        device = f"cuda:{actual_device_index}"
                        embedding_model.to(device)
                        if caption_model:
                            caption_model.to(device)
                        # Verify move was successful
                        new_device = next(embedding_model.parameters()).device
                        if new_device.type == 'cuda':
                            print(f"  ✓ Models successfully moved to {device}")
                            actual_device_type = 'cuda'
                        else:
                            print(f"  ✗ Failed to move models to GPU (still on {new_device})")
                    except Exception as e:
                        print(f"  ✗ Failed to move models to GPU: {e}")
                        print("  Continuing with CPU (will be slow)")
                
                # Display final status
                if actual_device_type == 'cuda':
                    gpu_name = torch.cuda.get_device_name(actual_device_index)
                    gpu_memory = torch.cuda.get_device_properties(actual_device_index).total_memory / (1024**3)
                    print("✅ GPU ACCELERATION ENABLED")
                    print(f"  Embedding Model Device: {embedding_device}")
                    if caption_model:
                        caption_device = next(caption_model.parameters()).device
                        print(f"  Caption Model Device: {caption_device}")
                    print(f"  GPU: {gpu_name}")
                    print(f"  GPU Memory: {gpu_memory:.2f} GB")
                    # Update device variable to match actual device
                    device = str(embedding_device)
                else:
                    print("❌ RUNNING ON CPU - VERY SLOW!")
                    print(f"  Device: {embedding_device}")
                    print(f"  CUDA Available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        print("  ⚠️  CUDA is available but models are on CPU!")
                        print("  This may be due to:")
                        print("    - PyTorch CPU-only installation")
                        print("    - GPU memory issues")
                        print("    - CUDA driver/runtime mismatch")
                    device = "cpu"
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not verify device: {e}")
                print(f"  Reported device: {device}")
            
            print("="*80 + "\n")
            
            # Extract audio
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video(video_path)
            full_audio_segment = None
            if audio_path and os.path.exists(audio_path):
                try:
                    full_audio_segment = AudioSegment.from_wav(audio_path)
                except Exception as e:
                    print(f"Warning: Could not load audio: {e}")
            
            # Split video into chunks
            print("Splitting video into chunks...")
            chunks = split_video_into_chunks(video_path)
            print(f"✓ Video split into {len(chunks)} chunk(s)")
            
            # Create upload managers
            print("Initializing upload managers...")
            video_upload_manager = PineconeUploadManager(video_index, num_workers=4, batch_size=100)
            audio_upload_manager = PineconeUploadManager(audio_index, num_workers=4, batch_size=100)
            text_upload_manager = PineconeUploadManager(text_index, num_workers=4, batch_size=100)
            desc_upload_manager = PineconeUploadManager(desc_index, num_workers=4, batch_size=100)
            
            # Process chunks
            print(f"\n🚀 Processing {len(chunks)} chunk(s)...")
            
            # Pre-detect scenes
            scene_counts = []
            for i, (start, end, chunk_path) in enumerate(chunks):
                count = len(detect_scenes_with_adaptive_detector(chunk_path))
                scene_counts.append(count)
            
            total_scenes = sum(scene_counts)
            print(f"✓ Total scenes detected: {total_scenes}")
            
            # Process chunks in parallel
            chunk_scene_offset = 0
            temp_audio_clips = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
                chunk_args_list = []
                
                for i, (chunk_start, chunk_end, chunk_path) in enumerate(chunks):
                    chunk_args = (
                        chunk_path, chunk_start, args.video_name, video_id, chunk_scene_offset,
                        transcript_segments, full_audio_segment,
                        video_upload_manager, audio_upload_manager, text_upload_manager, desc_upload_manager,
                        embedding_model, caption_processor, caption_model, device,
                        print
                    )
                    chunk_args_list.append(chunk_args)
                    chunk_scene_offset += scene_counts[i]
                
                futures = [executor.submit(process_video_chunk, arg) for arg in chunk_args_list]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    audio_files = future.result()
                    if audio_files:
                        temp_audio_clips.extend(audio_files)
                    print(f"✓ Chunk {i+1}/{len(chunks)} completed")
            
            # Wait for uploads
            print("\n📤 Finalizing uploads...")
            video_upload_manager.wait_for_completion()
            audio_upload_manager.wait_for_completion()
            text_upload_manager.wait_for_completion()
            desc_upload_manager.wait_for_completion()
            
            # Stop managers
            video_upload_manager.stop()
            audio_upload_manager.stop()
            text_upload_manager.stop()
            desc_upload_manager.stop()
            
            # Get stats
            video_stats = video_upload_manager.get_stats()
            audio_stats = audio_upload_manager.get_stats()
            text_stats = text_upload_manager.get_stats()
            desc_stats = desc_upload_manager.get_stats()
            
            print("\n" + "=" * 80)
            print("✅ EMBEDDING CREATION COMPLETE!")
            print("=" * 80)
            print(f"Video embeddings:    {video_stats['uploaded']} ({video_stats['errors']} errors)")
            print(f"Audio embeddings:    {audio_stats['uploaded']} ({audio_stats['errors']} errors)")
            print(f"Text embeddings:     {text_stats['uploaded']} ({text_stats['errors']} errors)")
            print(f"Description embeddings: {desc_stats['uploaded']} ({desc_stats['errors']} errors)")
            print(f"Total scenes: {total_scenes}")
            print("=" * 80)
            
            # Cleanup
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            for clip_path in temp_audio_clips:
                if os.path.exists(clip_path):
                    try:
                        os.remove(clip_path)
                    except:
                        pass
            
            return 0
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    else:
        # Both video and transcript are URLs - call process_video directly
        try:
            print("Processing with URLs...")
            process_video(
                videoPathURL=args.video_url,
                transcriptPathURL=args.transcript_url,
                video_name=args.video_name,
                video_id=video_id,
                status_callback=print
            )
            print("\n✅ Embedding creation complete!")
            return 0
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())

