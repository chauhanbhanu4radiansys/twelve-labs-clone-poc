"""
Main embedding pipeline that orchestrates video processing and embedding generation
"""
import os
import sys
import uuid
import tempfile
import concurrent.futures
import logging
import warnings
import cv2
import torch
from typing import Dict, List, Any, Tuple, Callable, Optional
from pydub import AudioSegment

# Suppress ImageBind logging warnings before any imports
warnings.filterwarnings('ignore', message='.*Large gap between audio.*')
logging.getLogger('imagebind.data').setLevel(logging.CRITICAL)
logging.getLogger('imagebind').setLevel(logging.CRITICAL)

from .config import PINECONE_CONFIG, BATCH_SIZE
from .models import load_imagebind_model, load_captioning_model
from .processing import (
    detect_scenes_with_adaptive_detector,
    split_video_into_chunks,
    prepare_scene_data,
    extract_audio_from_video
)
from .embeddings import get_batch_embeddings, generate_scene_descriptions
from .upload import PineconeUploadManager
from .download import download_from_url, download_transcript_from_url

# Import Pinecone
try:
    from pinecone import Pinecone
except ImportError:
    try:
        import pinecone
        Pinecone = pinecone.Pinecone
    except (ImportError, AttributeError):
        print("Error: Pinecone not available. Install with: pip install pinecone")
        Pinecone = None


def init_pinecone_indexes():
    """Initialize Pinecone indexes using credentials from environment variables."""
    # ============================================================================
    # Pinecone credentials are loaded from environment variables via config.py
    # Required: PINECONE_API_KEY
    # Optional: PINECONE_FRAME_INDEX, PINECONE_AUDIO_INDEX, PINECONE_TRANSCRIPT_INDEX, PINECONE_DESCRIPTION_INDEX
    # ============================================================================
    if Pinecone is None:
        return None, None, None, None
    
    try:
        pc = Pinecone(api_key=PINECONE_CONFIG['api_key'])
        
        index_list = pc.list_indexes().names()
        
        video_index = pc.Index(PINECONE_CONFIG['video_index_name']) if PINECONE_CONFIG['video_index_name'] in index_list else None
        audio_index = pc.Index(PINECONE_CONFIG['audio_index_name']) if PINECONE_CONFIG['audio_index_name'] in index_list else None
        text_index = pc.Index(PINECONE_CONFIG['text_index_name']) if PINECONE_CONFIG['text_index_name'] in index_list else None
        desc_index = pc.Index(PINECONE_CONFIG['desc_index_name']) if PINECONE_CONFIG['desc_index_name'] in index_list else None
        
        if not all([video_index, audio_index, text_index, desc_index]):
            print(f"Warning: Some Pinecone indexes not found. Available: {index_list}")
        
        return video_index, audio_index, text_index, desc_index
    except Exception as e:
        print(f"Failed to initialize Pinecone indexes: {e}")
        return None, None, None, None


def process_batch(
    batch_data: List[Dict[str, Any]],
    embedding_model,
    caption_processor,
    caption_model,
    device: str
) -> List[Dict[str, Any]]:
    """Processes a batch of scene data to generate descriptions and embeddings."""
    results = []
    pil_images = [item['pil_image'] for item in batch_data]
    audio_paths = [item['audio_path'] for item in batch_data]
    transcripts = [item['transcript'] for item in batch_data]

    # Batch generate descriptions
    descriptions = generate_scene_descriptions(pil_images, caption_processor, caption_model, device)

    # Batch generate embeddings for all modalities
    image_embeddings, audio_embeddings, text_embeddings = get_batch_embeddings(
        pil_images=pil_images, 
        audio_paths=audio_paths, 
        texts=transcripts,
        device=device, 
        model=embedding_model
    )
    
    _, _, desc_embeddings = get_batch_embeddings(
        pil_images=[], 
        audio_paths=[], 
        texts=descriptions,
        device=device, 
        model=embedding_model
    )

    for i, item in enumerate(batch_data):
        scene_uuid = str(uuid.uuid4())
        description = descriptions[i]
        metadata = {
            "video_name": item['video_name'], 
            "video_id": str(item['video_id']),
            "scene_index": item['scene_global_index'], 
            "start_time": item['abs_start_time'],
            "end_time": item['abs_end_time'], 
            "transcript": item['transcript'],
            "description": description, 
            "scene_uuid": scene_uuid
        }

        video_vector = None
        if image_embeddings.nelement() > 0 and i < len(image_embeddings):
            video_vector = {
                "id": scene_uuid, 
                "values": image_embeddings[i].cpu().numpy().tolist(), 
                "metadata": metadata
            }

        audio_vector = None
        if audio_embeddings.nelement() > 0 and i < len(audio_embeddings):
            audio_vector = {
                "id": scene_uuid, 
                "values": audio_embeddings[i].cpu().numpy().tolist(), 
                "metadata": metadata
            }
            
        text_vector = None
        if text_embeddings.nelement() > 0 and i < len(text_embeddings):
            text_vector = {
                "id": scene_uuid, 
                "values": text_embeddings[i].cpu().numpy().tolist(), 
                "metadata": metadata
            }

        desc_vector = None
        if desc_embeddings.nelement() > 0 and i < len(desc_embeddings):
            desc_vector = {
                "id": scene_uuid, 
                "values": desc_embeddings[i].cpu().numpy().tolist(), 
                "metadata": metadata
            }
        
        results.append({
            "video_vector": video_vector,
            "audio_vector": audio_vector,
            "text_vector": text_vector,
            "desc_vector": desc_vector
        })
        
    return results


def process_video_chunk(
    args: Tuple
) -> Tuple[List[str]]:
    """
    Processes a single video chunk to detect scenes, prepare data, and generate embeddings.
    """
    (
        chunk_path,
        chunk_start_offset,
        video_name,
        video_id,
        chunk_scene_offset,
        transcript_segments,
        full_audio_segment,
        video_upload_manager,
        audio_upload_manager,
        text_upload_manager,
        desc_upload_manager,
        embedding_model,
        caption_processor,
        caption_model,
        device,
        log_callback
    ) = args

    chunk_name = os.path.basename(chunk_path)
    chunk_duration = chunk_start_offset
    log_callback(f"\n{'='*80}")
    log_callback(f"📹 Processing Chunk: {chunk_name} (starts at {chunk_duration:.2f}s)")
    log_callback(f"{'='*80}")

    # Detect scenes for this chunk
    log_callback(f"  🔍 Detecting scenes in chunk...")
    scene_list = detect_scenes_with_adaptive_detector(chunk_path)
    if not scene_list:
        log_callback(f"  ⚠ No scenes detected in chunk {chunk_name}")
        return []
    
    log_callback(f"  ✓ Detected {len(scene_list)} scenes in chunk")

    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Prepare data for all scenes in this chunk in parallel
    log_callback(f"  📦 Preparing scene data (extracting frames, audio clips, transcripts)...")
    scene_args = [
        (chunk_path, scene, fps, chunk_start_offset, video_name, video_id, chunk_scene_offset + i, transcript_segments, full_audio_segment)
        for i, scene in enumerate(scene_list)
    ]
    
    chunk_scene_data = []
    temp_audio_clips = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(prepare_scene_data, scene_args)
        for scene_data in results:
            if scene_data:
                chunk_scene_data.append(scene_data)
                if scene_data["audio_path"]:
                    temp_audio_clips.append(scene_data["audio_path"])
    
    log_callback(f"  ✓ Prepared {len(chunk_scene_data)} scenes for processing")

    # Sort data by scene index to maintain chronological order
    chunk_scene_data.sort(key=lambda x: x['scene_global_index'])

    # Process the prepared scene data in batches
    num_batches = (len(chunk_scene_data) + BATCH_SIZE - 1) // BATCH_SIZE
    log_callback(f"  🧠 Generating embeddings in {num_batches} batch(es) of up to {BATCH_SIZE} scenes each...")
    
    # Batch size for immediate queuing (smaller batches, upload faster - matches reference speed)
    IMMEDIATE_BATCH_SIZE = 20  # Queue every 20 vectors instead of waiting for 100
    
    # Accumulate vectors for immediate small-batch uploads
    video_vectors_batch = []
    audio_vectors_batch = []
    text_vectors_batch = []
    desc_vectors_batch = []
    
    total_vectors_queued = 0
    
    for i in range(0, len(chunk_scene_data), BATCH_SIZE):
        batch = chunk_scene_data[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        log_callback(f"    📊 Batch {batch_num}/{num_batches}: Processing {len(batch)} scenes...")
        
        batch_results = process_batch(batch, embedding_model, caption_processor, caption_model, device)

        # Collect vectors and upload immediately in small batches (parallel with processing)
        for result in batch_results:
            if result['video_vector']:
                video_vectors_batch.append(result['video_vector'])
                # Upload immediately when small batch is ready
                if len(video_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    video_upload_manager.add_to_queue(video_vectors_batch)
                    total_vectors_queued += len(video_vectors_batch)
                    log_callback(f"      📤 Video: Queued batch with {len(video_vectors_batch)} embeddings")
                    video_vectors_batch = []
            
            if result['audio_vector']:
                audio_vectors_batch.append(result['audio_vector'])
                if len(audio_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    audio_upload_manager.add_to_queue(audio_vectors_batch)
                    total_vectors_queued += len(audio_vectors_batch)
                    log_callback(f"      📤 Audio: Queued batch with {len(audio_vectors_batch)} embeddings")
                    audio_vectors_batch = []
            
            if result['text_vector']:
                text_vectors_batch.append(result['text_vector'])
                if len(text_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    text_upload_manager.add_to_queue(text_vectors_batch)
                    total_vectors_queued += len(text_vectors_batch)
                    log_callback(f"      📤 Text: Queued batch with {len(text_vectors_batch)} embeddings")
                    text_vectors_batch = []
            
            if result['desc_vector']:
                desc_vectors_batch.append(result['desc_vector'])
                if len(desc_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    desc_upload_manager.add_to_queue(desc_vectors_batch)
                    total_vectors_queued += len(desc_vectors_batch)
                    log_callback(f"      📤 Description: Queued batch with {len(desc_vectors_batch)} embeddings")
                    desc_vectors_batch = []
        
        log_callback(f"    ✓ Batch {batch_num}/{num_batches}: Processed {len(batch)} scenes ({total_vectors_queued} vectors queued so far)")
    
    # Upload any remaining vectors (final flush)
    if video_vectors_batch:
        video_upload_manager.add_to_queue(video_vectors_batch)
        total_vectors_queued += len(video_vectors_batch)
        log_callback(f"      📤 Video: Queued final batch with {len(video_vectors_batch)} embeddings")
    if audio_vectors_batch:
        audio_upload_manager.add_to_queue(audio_vectors_batch)
        total_vectors_queued += len(audio_vectors_batch)
        log_callback(f"      📤 Audio: Queued final batch with {len(audio_vectors_batch)} embeddings")
    if text_vectors_batch:
        text_upload_manager.add_to_queue(text_vectors_batch)
        total_vectors_queued += len(text_vectors_batch)
        log_callback(f"      📤 Text: Queued final batch with {len(text_vectors_batch)} embeddings")
    if desc_vectors_batch:
        desc_upload_manager.add_to_queue(desc_vectors_batch)
        total_vectors_queued += len(desc_vectors_batch)
        log_callback(f"      📤 Description: Queued final batch with {len(desc_vectors_batch)} embeddings")
    
    log_callback(f"  ✅ Chunk {chunk_name} processing complete: {len(chunk_scene_data)} scenes, {total_vectors_queued} vectors queued")
    log_callback(f"{'='*80}\n")

    return temp_audio_clips


def process_video(
    videoPathURL: str,
    transcriptPathURL: str,
    video_name: str,
    video_id: str = None,
    status_callback: Callable[[str], None] = None
):
    """
    Main processing pipeline for a single video.
    
    Args:
        videoPathURL: Pre-signed S3 URL to video file
        transcriptPathURL: Pre-signed S3 URL to transcript JSON file
        video_name: Name of the video
        video_id: Unique identifier for the video (generated if None)
        status_callback: Optional callback function for status updates
    """
    if status_callback is None:
        status_callback = print
    
    if video_id is None:
        video_id = str(uuid.uuid4())
    
    video_path = None
    audio_path = None
    temp_audio_clips = []
    
    try:
        # Initialize Pinecone clients
        status_callback("Initializing Pinecone clients...")
        video_index, audio_index, text_index, desc_index = init_pinecone_indexes()
        
        if not all([video_index, audio_index, text_index, desc_index]):
            raise Exception("Failed to initialize Pinecone indexes. Check credentials in config.py")
        
        # Download video from pre-signed URL
        status_callback("Downloading video from URL...")
        video_path = download_from_url(videoPathURL, progress_callback=status_callback)
        if not video_path or not os.path.exists(video_path):
            raise Exception(f"Failed to download video from URL: {videoPathURL[:100]}...")
        status_callback(f"Video downloaded to temporary file: {video_path}")
        
        # Download transcript from pre-signed URL
        status_callback("Downloading transcript from URL...")
        transcript_segments = download_transcript_from_url(transcriptPathURL)
        if not transcript_segments:
            status_callback("Warning: Could not download transcript. Audio processing will be skipped.")
        
        # Load models
        status_callback("Loading models...")
        embedding_model, device = load_imagebind_model()
        caption_processor, caption_model = load_captioning_model()
        
        # VERIFY GPU USAGE - Critical for performance
        status_callback("\n" + "="*80)
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    status_callback("✅ GPU ACCELERATION ENABLED")
                    status_callback(f"  Device: {device}")
                    status_callback(f"  GPU: {gpu_name}")
                    status_callback(f"  GPU Memory: {gpu_memory:.2f} GB")
                    status_callback(f"  CUDA Available: {torch.cuda.is_available()}")
                except Exception as e:
                    status_callback(f"⚠️  Warning: Could not get GPU details: {e}")
                    status_callback(f"  Device: {device}")
            else:
                status_callback(f"⚠️  WARNING: Device set to {device} but CUDA not available!")
        else:
            status_callback("❌ CRITICAL WARNING: RUNNING ON CPU - VERY SLOW!")
            status_callback(f"  Device: {device}")
            status_callback(f"  CUDA Available: {torch.cuda.is_available()}")
            status_callback(f"  This will be 10-100x slower than GPU")
            status_callback(f"  Expected processing time: 10-30+ minutes (vs 1-2 minutes on GPU)")
            status_callback("")
            status_callback("  To enable GPU:")
            status_callback("  1. Ensure Docker has GPU access: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi")
            status_callback("  2. Install NVIDIA Container Toolkit if not installed")
            status_callback("  3. Restart Docker: sudo systemctl restart docker")
        status_callback("="*80 + "\n")
        
        if not embedding_model:
            raise Exception("Failed to load ImageBind model")
        
        # Extract audio from video for scene audio slicing
        status_callback("Extracting audio from video...")
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            status_callback("Warning: Could not extract audio. Audio embeddings will be skipped.")
        
        # Load full audio for slicing
        full_audio_segment = None
        if audio_path and os.path.exists(audio_path):
            try:
                full_audio_segment = AudioSegment.from_wav(audio_path)
            except Exception as e:
                status_callback(f"Warning: Could not load audio for slicing: {e}")

        # Split video into chunks
        status_callback("Splitting video into chunks...")
        chunks = split_video_into_chunks(video_path)
        is_chunked = len(chunks) > 1 and chunks[0][2] != video_path
        
        if is_chunked:
            status_callback(f"📹 Video split into {len(chunks)} chunks for parallel processing")
            for i, (start, end, path) in enumerate(chunks):
                status_callback(f"  Chunk {i+1}: {start:.2f}s - {end:.2f}s ({os.path.basename(path)})")
        else:
            status_callback(f"📹 Processing single video file (no chunking needed)")
        
        status_callback(f"\n🚀 Starting parallel processing of {len(chunks)} chunk(s)...")
        
        # Create shared upload managers with logging (matching reference: 4 workers)
        status_callback("📤 Initializing Pinecone upload managers...")
        video_upload_manager = PineconeUploadManager(video_index, num_workers=4, batch_size=100, log_callback=status_callback)
        audio_upload_manager = PineconeUploadManager(audio_index, num_workers=4, batch_size=100, log_callback=status_callback)
        text_upload_manager = PineconeUploadManager(text_index, num_workers=4, batch_size=100, log_callback=status_callback)
        desc_upload_manager = PineconeUploadManager(desc_index, num_workers=4, batch_size=100, log_callback=status_callback)

        # Pre-calculate scene counts
        status_callback("🔍 Pre-detecting scenes in all chunks...")
        scene_counts = []
        for i, (start, end, chunk_path) in enumerate(chunks):
            count = len(detect_scenes_with_adaptive_detector(chunk_path))
            scene_counts.append(count)
            status_callback(f"  Chunk {i+1}: {count} scenes detected")
        
        total_scenes = sum(scene_counts)
        status_callback(f"✓ Total scenes detected: {total_scenes} across {len(chunks)} chunk(s)")
        status_callback(f"🔄 Starting parallel chunk processing with {min(len(chunks), 4)} workers...\n")

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            chunk_args_list = []
            chunk_scene_offset = 0
            
            for i, (chunk_start, chunk_end, chunk_path) in enumerate(chunks):
                args = (
                    chunk_path, chunk_start, video_name, video_id, chunk_scene_offset,
                    transcript_segments, full_audio_segment,
                    video_upload_manager, audio_upload_manager, text_upload_manager, desc_upload_manager,
                    embedding_model, caption_processor, caption_model, device,
                    status_callback
                )
                chunk_args_list.append(args)
                chunk_scene_offset += scene_counts[i]
            
            # Submit chunk processing tasks
            future_to_chunk = {executor.submit(process_video_chunk, arg): (i, arg) for i, arg in enumerate(chunk_args_list)}
            
            processed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx, _ = future_to_chunk[future]
                audio_files = future.result()
                if audio_files:
                    temp_audio_clips.extend(audio_files)
                processed_chunks += 1
                status_callback(f"\n✅ Chunk {chunk_idx + 1}/{len(chunks)} completed ({processed_chunks}/{len(chunks)} total)")

        # Note: Transcript and video are already in S3, accessed via pre-signed URLs
        # No need to upload anything to MinIO

        # Wait for all uploads to complete
        status_callback("\n" + "="*80)
        status_callback("📤 Finalizing Pinecone uploads...")
        status_callback("="*80)
        
        status_callback("\n📹 Video embeddings:")
        video_upload_manager.wait_for_completion()
        
        status_callback("\n🔊 Audio embeddings:")
        audio_upload_manager.wait_for_completion()
        
        status_callback("\n📝 Text embeddings:")
        text_upload_manager.wait_for_completion()
        
        status_callback("\n📄 Description embeddings:")
        desc_upload_manager.wait_for_completion()
        
        # Stop upload managers
        video_upload_manager.stop()
        audio_upload_manager.stop()
        text_upload_manager.stop()
        desc_upload_manager.stop()
        
        # Get stats
        video_stats = video_upload_manager.get_stats()
        audio_stats = audio_upload_manager.get_stats()
        text_stats = text_upload_manager.get_stats()
        desc_stats = desc_upload_manager.get_stats()
        
        status_callback("\n" + "="*80)
        status_callback("✅ PROCESSING COMPLETE!")
        status_callback("="*80)
        status_callback(f"📊 Upload Statistics:")
        status_callback(f"  • Video embeddings:    {video_stats['uploaded']} embeddings ({video_stats['errors']} errors)")
        status_callback(f"  • Audio embeddings:    {audio_stats['uploaded']} embeddings ({audio_stats['errors']} errors)")
        status_callback(f"  • Text embeddings:     {text_stats['uploaded']} embeddings ({text_stats['errors']} errors)")
        status_callback(f"  • Description embeddings: {desc_stats['uploaded']} embeddings ({desc_stats['errors']} errors)")
        status_callback(f"  • Total scenes processed: {total_scenes}")
        total_embeddings = video_stats['uploaded'] + audio_stats['uploaded'] + text_stats['uploaded'] + desc_stats['uploaded']
        status_callback(f"  • Total embeddings uploaded: {total_embeddings}")
        status_callback(f"  • Average embeddings per scene: {total_embeddings / total_scenes if total_scenes > 0 else 0:.1f}")
        status_callback("="*80 + "\n")
        
        # Cleanup chunk files
        if is_chunked:
            for _, _, chunk_path in chunks:
                if video_path != chunk_path:
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass
            try:
                os.rmdir(os.path.dirname(chunks[0][2]))
            except OSError:
                pass

    except Exception as e:
        status_callback(f"Processing failed: {e}")
        raise
    finally:
        # Cleanup downloaded video file - suppress all exceptions to prevent propagation
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                # Silently ignore cleanup errors - don't let them propagate
                try:
                    if status_callback:
                        status_callback(f"⚠ Warning: Error cleaning up downloaded video file: {e}")
                except:
                    pass  # Even status_callback might fail if stderr is closed
        
        # Cleanup audio files - suppress all exceptions
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                try:
                    if status_callback:
                        status_callback(f"⚠ Warning: Error cleaning up main audio file")
                except:
                    pass
        
        # Cleanup audio clips - suppress all exceptions
        for clip_path in temp_audio_clips:
            if os.path.exists(clip_path):
                try:
                    os.remove(clip_path)
                except Exception:
                    pass  # Silently ignore cleanup errors

