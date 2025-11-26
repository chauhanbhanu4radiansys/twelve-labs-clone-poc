"""
Main embedding pipeline that orchestrates video processing and embedding generation
"""
import os
import sys
import uuid
import tempfile
import concurrent.futures
import cv2
from typing import Dict, List, Any, Tuple, Callable, Optional
from pydub import AudioSegment

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
    """Initialize Pinecone indexes with hardcoded credentials."""
    # ============================================================================
    # PLACEHOLDER: Pinecone credentials are in src/embedding/config.py
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
        device
    ) = args

    # Detect scenes for this chunk
    scene_list = detect_scenes_with_adaptive_detector(chunk_path)
    if not scene_list:
        return []

    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Prepare data for all scenes in this chunk in parallel
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

    # Sort data by scene index to maintain chronological order
    chunk_scene_data.sort(key=lambda x: x['scene_global_index'])

    # Process the prepared scene data in batches
    for i in range(0, len(chunk_scene_data), BATCH_SIZE):
        batch = chunk_scene_data[i:i + BATCH_SIZE]
        batch_results = process_batch(batch, embedding_model, caption_processor, caption_model, device)

        for result in batch_results:
            if result['video_vector']:
                video_upload_manager.add_to_queue([result['video_vector']])
            if result['audio_vector']:
                audio_upload_manager.add_to_queue([result['audio_vector']])
            if result['text_vector']:
                text_upload_manager.add_to_queue([result['text_vector']])
            if result['desc_vector']:
                desc_upload_manager.add_to_queue([result['desc_vector']])

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
        
        status_callback(f"Processing {len(chunks)} video chunk(s) in parallel...")
        
        # Create shared upload managers
        video_upload_manager = PineconeUploadManager(video_index)
        audio_upload_manager = PineconeUploadManager(audio_index)
        text_upload_manager = PineconeUploadManager(text_index)
        desc_upload_manager = PineconeUploadManager(desc_index)

        # Pre-calculate scene counts
        scene_counts = [len(detect_scenes_with_adaptive_detector(c[2])) for c in chunks]
        total_scenes = sum(scene_counts)
        status_callback(f"Detected {total_scenes} scenes. Starting parallel processing...")

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            chunk_args_list = []
            chunk_scene_offset = 0
            
            for i, (chunk_start, _, chunk_path) in enumerate(chunks):
                args = (
                    chunk_path, chunk_start, video_name, video_id, chunk_scene_offset,
                    transcript_segments, full_audio_segment,
                    video_upload_manager, audio_upload_manager, text_upload_manager, desc_upload_manager,
                    embedding_model, caption_processor, caption_model, device
                )
                chunk_args_list.append(args)
                chunk_scene_offset += scene_counts[i]
            
            # Submit chunk processing tasks
            future_to_chunk = {executor.submit(process_video_chunk, arg): arg for arg in chunk_args_list}
            
            processed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                audio_files = future.result()
                if audio_files:
                    temp_audio_clips.extend(audio_files)
                processed_chunks += 1
                status_callback(f"Completed processing chunk {processed_chunks}/{len(chunks)}...")

        # Note: Transcript and video are already in S3, accessed via pre-signed URLs
        # No need to upload anything to MinIO

        # Wait for all uploads to complete
        status_callback("Waiting for Pinecone uploads to complete...")
        video_upload_manager.wait_for_completion()
        audio_upload_manager.wait_for_completion()
        text_upload_manager.wait_for_completion()
        desc_upload_manager.wait_for_completion()
        
        video_upload_manager.stop()
        audio_upload_manager.stop()
        text_upload_manager.stop()
        desc_upload_manager.stop()
        
        # Get stats
        video_stats = video_upload_manager.get_stats()
        audio_stats = audio_upload_manager.get_stats()
        text_stats = text_upload_manager.get_stats()
        desc_stats = desc_upload_manager.get_stats()
        
        status_callback(
            f"Processing complete! "
            f"{video_stats['uploaded']} video, "
            f"{audio_stats['uploaded']} audio, "
            f"{text_stats['uploaded']} text, "
            f"{desc_stats['uploaded']} description embeddings uploaded."
        )
        
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
        # Cleanup downloaded video file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as e:
                print(f"Error cleaning up downloaded video file: {e}", file=sys.stderr)
        
        # Cleanup audio files
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError as e:
                print(f"Error cleaning up main audio file: {e}", file=sys.stderr)
        
        for clip_path in temp_audio_clips:
            if os.path.exists(clip_path):
                try:
                    os.remove(clip_path)
                except OSError as e:
                    print(f"Error cleaning up audio clip {clip_path}: {e}", file=sys.stderr)

