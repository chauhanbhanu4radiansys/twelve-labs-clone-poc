#!/usr/bin/env python3
"""
Pinecone Search Script
Searches Pinecone indexes using ImageBind embeddings for text, audio, video, or image inputs.

Usage:
    python pinecone_search.py text "your query" [top_k]
    python pinecone_search.py image /path/to/image.jpg [top_k]
    python pinecone_search.py audio /path/to/audio.wav [top_k]
    python pinecone_search.py video /path/to/video.mp4 [top_k]
    
Environment Variables (from .env):
    PINECONE_API_KEY - Required
    PINECONE_FRAME_INDEX - Video index name (default: video-search)
    PINECONE_AUDIO_INDEX - Audio index name (default: audio-search)
    PINECONE_TRANSCRIPT_INDEX - Text index name (default: text-search)
    PINECONE_DESCRIPTION_INDEX - Description index name (default: desc-search)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
import torch
from PIL import Image
import cv2

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

# Import project modules
from src.embedding.config import PINECONE_CONFIG
from src.embedding.models import load_imagebind_model, load_captioning_model, load_whisper_model
from src.embedding.pipeline import init_pinecone_indexes
from src.embedding.embeddings import get_batch_embeddings, generate_scene_descriptions
from src.retrieval.search import query_index
from src.retrieval.scoring import compute_final_score, compute_audio_search_score
from src.retrieval.utils import merge_overlapping_clips


def extract_frame_from_video(video_path: str, timestamp: float = 0.0) -> Image.Image:
    """
    Extract a single frame from video at given timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds (default: 0.0 for first frame)
        
    Returns:
        PIL Image
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set video position to timestamp
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        # Fallback to first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame from video: {video_path}")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def search_text(
    query_text: str,
    video_index,
    audio_index,
    text_index,
    desc_index,
    embedding_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using text query across all indexes.
    
    Args:
        query_text: Text query string
        video_index, audio_index, text_index, desc_index: Pinecone indexes
        embedding_model: ImageBind model
        device: Device to run model on
        top_k: Number of results per index
        filter_dict: Optional metadata filter
        
    Returns:
        List of search results
    """
    print(f"Generating text embedding for: '{query_text}'")
    
    # Generate text embedding
    _, _, text_embeddings = get_batch_embeddings(
        pil_images=[],
        audio_paths=[],
        texts=[query_text],
        device=device,
        model=embedding_model
    )
    
    if text_embeddings.nelement() == 0:
        print("Error: Failed to generate text embedding")
        return []
    
    query_vector = text_embeddings[0].cpu().numpy().tolist()
    
    # Query all indexes in parallel
    print("Querying all indexes...")
    results_map = {}
    indexes_to_query = {
        'video': video_index,
        'audio': audio_index,
        'text': text_index,
        'desc': desc_index
    }
    
    for name, index in indexes_to_query.items():
        if index is None:
            continue
        try:
            results = query_index(index, query_vector, top_k, filter_dict)
            results_map[name] = results
            if results and hasattr(results, 'matches'):
                print(f"  {name}: {len(results.matches)} results")
        except Exception as e:
            print(f"  Error querying {name} index: {e}")
            results_map[name] = None
    
    # Merge results by scene_uuid
    merged_results = {}
    for source_name, results in results_map.items():
        if not results or not hasattr(results, 'matches'):
            continue
        
        for match in results.matches:
            scene_uuid = match.metadata.get("scene_uuid")
            if not scene_uuid:
                continue
            
            if scene_uuid not in merged_results:
                merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
            else:
                merged_results[scene_uuid]['scores'][source_name] = match.score
                if match.score > merged_results[scene_uuid]['match'].score:
                    merged_results[scene_uuid]['match'] = match
    
    # Compute final scores
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def search_image(
    image_path: str,
    video_index,
    desc_index,
    embedding_model,
    caption_processor,
    caption_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using image query.
    
    Args:
        image_path: Path to image file
        video_index, desc_index: Pinecone indexes
        embedding_model: ImageBind model
        caption_processor, caption_model: BLIP models for captioning
        device: Device to run model on
        top_k: Number of results per index
        filter_dict: Optional metadata filter
        
    Returns:
        List of search results
    """
    print(f"Loading image: {image_path}")
    
    # Load image
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return []
    
    # Generate image embedding
    print("Generating image embedding...")
    image_embeddings, _, _ = get_batch_embeddings(
        pil_images=[pil_image],
        audio_paths=[],
        texts=[],
        device=device,
        model=embedding_model
    )
    
    search_tasks = {}
    if image_embeddings.nelement() > 0:
        search_tasks['video'] = (video_index, image_embeddings[0].cpu().numpy().tolist())
    
    # Generate description and its embedding
    if caption_processor and caption_model:
        print("Generating image caption...")
        desc = generate_scene_descriptions([pil_image], caption_processor, caption_model, device)[0]
        print(f"Caption: {desc}")
        
        if desc:
            _, _, desc_embeddings = get_batch_embeddings(
                pil_images=[],
                audio_paths=[],
                texts=[desc],
                device=device,
                model=embedding_model
            )
            if desc_embeddings.nelement() > 0:
                search_tasks['desc'] = (desc_index, desc_embeddings[0].cpu().numpy().tolist())
    
    if not search_tasks:
        print("Error: Failed to generate embeddings")
        return []
    
    # Query indexes
    print("Querying indexes...")
    results_map = {}
    for name, (index, query_vector) in search_tasks.items():
        if index is None:
            continue
        try:
            results = query_index(index, query_vector, top_k, filter_dict)
            results_map[name] = results
            if results and hasattr(results, 'matches'):
                print(f"  {name}: {len(results.matches)} results")
        except Exception as e:
            print(f"  Error querying {name} index: {e}")
            results_map[name] = None
    
    # Merge results
    merged_results = {}
    for source_name, results in results_map.items():
        if not results or not hasattr(results, 'matches'):
            continue
        
        for match in results.matches:
            scene_uuid = match.metadata.get("scene_uuid")
            if not scene_uuid:
                continue
            
            if scene_uuid not in merged_results:
                merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
            else:
                merged_results[scene_uuid]['scores'][source_name] = match.score
                if match.score > merged_results[scene_uuid]['match'].score:
                    merged_results[scene_uuid]['match'] = match
    
    # Compute final scores
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def search_audio(
    audio_path: str,
    audio_index,
    text_index,
    embedding_model,
    whisper_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using audio query.
    
    Args:
        audio_path: Path to audio file
        audio_index, text_index: Pinecone indexes
        embedding_model: ImageBind model
        whisper_model: Whisper model for transcription
        device: Device to run model on
        top_k: Number of results per index
        filter_dict: Optional metadata filter
        
    Returns:
        List of search results
    """
    print(f"Loading audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return []
    
    # Generate audio embedding
    print("Generating audio embedding...")
    _, audio_embeddings, _ = get_batch_embeddings(
        pil_images=[],
        audio_paths=[audio_path],
        texts=[],
        device=device,
        model=embedding_model
    )
    
    search_tasks = {}
    if audio_embeddings.nelement() > 0:
        search_tasks['audio'] = (audio_index, audio_embeddings[0].cpu().numpy().tolist())
    
    # Transcribe audio and generate text embedding
    transcript = None
    if whisper_model:
        try:
            print("Transcribing audio...")
            result = whisper_model.transcribe(
                audio_path,
                fp16=torch.cuda.is_available(),
                task='translate'
            )
            transcript = result.get("text", "").strip()
            print(f"Transcript: {transcript}")
        except Exception as e:
            print(f"Error transcribing audio: {e}")
    
    if transcript:
        _, _, text_embeddings = get_batch_embeddings(
            pil_images=[],
            audio_paths=[],
            texts=[transcript],
            device=device,
            model=embedding_model
        )
        if text_embeddings.nelement() > 0:
            search_tasks['text'] = (text_index, text_embeddings[0].cpu().numpy().tolist())
    
    if not search_tasks:
        print("Error: Failed to generate embeddings")
        return []
    
    # Query indexes
    print("Querying indexes...")
    results_map = {}
    for name, (index, query_vector) in search_tasks.items():
        if index is None:
            continue
        try:
            results = query_index(index, query_vector, top_k, filter_dict)
            results_map[name] = results
            if results and hasattr(results, 'matches'):
                print(f"  {name}: {len(results.matches)} results")
        except Exception as e:
            print(f"  Error querying {name} index: {e}")
            results_map[name] = None
    
    # Merge results
    merged_results = {}
    for source_name, results in results_map.items():
        if not results or not hasattr(results, 'matches'):
            continue
        
        for match in results.matches:
            scene_uuid = match.metadata.get("scene_uuid")
            if not scene_uuid:
                continue
            
            if scene_uuid not in merged_results:
                merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
            else:
                merged_results[scene_uuid]['scores'][source_name] = match.score
                if match.score > merged_results[scene_uuid]['match'].score:
                    merged_results[scene_uuid]['match'] = match
    
    # Compute final scores using audio-specific scoring
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        best_match.score = compute_audio_search_score(scores_dict)
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def search_video(
    video_path: str,
    video_index,
    desc_index,
    embedding_model,
    caption_processor,
    caption_model,
    device: str,
    timestamp: float = 0.0,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using video query (extracts frame and searches).
    
    Args:
        video_path: Path to video file
        video_index, desc_index: Pinecone indexes
        embedding_model: ImageBind model
        caption_processor, caption_model: BLIP models for captioning
        device: Device to run model on
        timestamp: Timestamp in seconds to extract frame (default: 0.0)
        top_k: Number of results per index
        filter_dict: Optional metadata filter
        
    Returns:
        List of search results
    """
    print(f"Extracting frame from video: {video_path} at {timestamp}s")
    
    try:
        pil_image = extract_frame_from_video(video_path, timestamp)
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return []
    
    # Generate image embedding
    print("Generating image embedding from video frame...")
    image_embeddings, _, _ = get_batch_embeddings(
        pil_images=[pil_image],
        audio_paths=[],
        texts=[],
        device=device,
        model=embedding_model
    )
    
    search_tasks = {}
    if image_embeddings.nelement() > 0:
        search_tasks['video'] = (video_index, image_embeddings[0].cpu().numpy().tolist())
    
    # Generate description and its embedding
    if caption_processor and caption_model:
        print("Generating frame caption...")
        desc = generate_scene_descriptions([pil_image], caption_processor, caption_model, device)[0]
        print(f"Caption: {desc}")
        
        if desc:
            _, _, desc_embeddings = get_batch_embeddings(
                pil_images=[],
                audio_paths=[],
                texts=[desc],
                device=device,
                model=embedding_model
            )
            if desc_embeddings.nelement() > 0:
                search_tasks['desc'] = (desc_index, desc_embeddings[0].cpu().numpy().tolist())
    
    if not search_tasks:
        print("Error: Failed to generate embeddings")
        return []
    
    # Query indexes
    print("Querying indexes...")
    results_map = {}
    for name, (index, query_vector) in search_tasks.items():
        if index is None:
            continue
        try:
            results = query_index(index, query_vector, top_k, filter_dict)
            results_map[name] = results
            if results and hasattr(results, 'matches'):
                print(f"  {name}: {len(results.matches)} results")
        except Exception as e:
            print(f"  Error querying {name} index: {e}")
            results_map[name] = None
    
    # Merge results
    merged_results = {}
    for source_name, results in results_map.items():
        if not results or not hasattr(results, 'matches'):
            continue
        
        for match in results.matches:
            scene_uuid = match.metadata.get("scene_uuid")
            if not scene_uuid:
                continue
            
            if scene_uuid not in merged_results:
                merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
            else:
                merged_results[scene_uuid]['scores'][source_name] = match.score
                if match.score > merged_results[scene_uuid]['match'].score:
                    merged_results[scene_uuid]['match'] = match
    
    # Compute final scores
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def format_results(results: List[Any]) -> List[Dict]:
    """Format search results for output."""
    formatted = []
    for result in results:
        if hasattr(result, 'metadata'):
            metadata = dict(result.metadata)
        else:
            metadata = {}
        
        score = float(result.score) if hasattr(result, 'score') else 0.0
        
        # Format time range
        start_time = metadata.get('start_time', 0)
        end_time = metadata.get('end_time', 0)
        
        def format_timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        start_timestamp = format_timestamp(start_time)
        end_timestamp = format_timestamp(end_time)
        time_range = f"{start_timestamp} - {end_timestamp}"
        
        formatted.append({
            'score': score,
            'video_name': metadata.get('video_name', 'N/A'),
            'video_id': metadata.get('video_id', 'N/A'),
            'start_time': start_time,
            'end_time': end_time,
            'time_range': time_range,
            'transcript': metadata.get('transcript', ''),
            'description': metadata.get('description', ''),
            'source': metadata.get('source', ''),
            'scene_uuid': metadata.get('scene_uuid', ''),
        })
    
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description='Search Pinecone indexes using ImageBind embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pinecone_search.py text "a person running" 10
  python pinecone_search.py image /path/to/image.jpg 8
  python pinecone_search.py audio /path/to/audio.wav 10
  python pinecone_search.py video /path/to/video.mp4 8 --timestamp 30.5
        """
    )
    
    parser.add_argument('input_type', choices=['text', 'image', 'audio', 'video'],
                        help='Type of input: text, image, audio, or video')
    parser.add_argument('input_value', 
                        help='Input value: text string, or path to image/audio/video file')
    parser.add_argument('top_k', type=int, nargs='?', default=8,
                        help='Number of results to return (default: 8)')
    parser.add_argument('--timestamp', type=float, default=0.0,
                        help='Timestamp in seconds for video frame extraction (default: 0.0)')
    parser.add_argument('--filter', type=str,
                        help='JSON string for metadata filter (e.g., \'{"video_name": "video.mp4"}\')')
    parser.add_argument('--output', type=str,
                        help='Output file path for JSON results (optional)')
    
    args = parser.parse_args()
    
    # Parse filter if provided
    filter_dict = None
    if args.filter:
        try:
            filter_dict = json.loads(args.filter)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON filter: {args.filter}")
            return 1
    
    # Validate input file exists (for non-text inputs)
    if args.input_type != 'text':
        if not os.path.exists(args.input_value):
            print(f"Error: File not found: {args.input_value}")
            return 1
    
    # Check environment variables
    if not PINECONE_CONFIG['api_key']:
        print("Error: PINECONE_API_KEY not found in .env file")
        return 1
    
    print("=" * 80)
    print("PINECONE SEARCH")
    print("=" * 80)
    print(f"Input Type: {args.input_type}")
    print(f"Input Value: {args.input_value}")
    print(f"Top K: {args.top_k}")
    if filter_dict:
        print(f"Filter: {filter_dict}")
    print("=" * 80)
    print()
    
    # Initialize Pinecone indexes
    print("Initializing Pinecone indexes...")
    video_index, audio_index, text_index, desc_index = init_pinecone_indexes()
    
    if not all([video_index, audio_index, text_index, desc_index]):
        print("Warning: Some indexes not found. Available indexes:")
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_CONFIG['api_key'])
        print(f"  {pc.list_indexes().names()}")
        return 1
    
    print("✓ Indexes initialized")
    print()
    
    # Load models
    print("Loading ImageBind model...")
    embedding_model, device = load_imagebind_model()
    print(f"✓ ImageBind loaded on {device}")
    
    caption_processor = None
    caption_model = None
    whisper_model = None
    
    if args.input_type in ['image', 'video']:
        print("Loading BLIP captioning model...")
        caption_processor, caption_model = load_captioning_model()
        if caption_processor and caption_model:
            print("✓ BLIP loaded")
    
    if args.input_type == 'audio':
        print("Loading Whisper model...")
        whisper_model = load_whisper_model()
        if whisper_model:
            print("✓ Whisper loaded")
    
    print()
    
    # Perform search
    try:
        if args.input_type == 'text':
            results = search_text(
                query_text=args.input_value,
                video_index=video_index,
                audio_index=audio_index,
                text_index=text_index,
                desc_index=desc_index,
                embedding_model=embedding_model,
                device=device,
                top_k=args.top_k,
                filter_dict=filter_dict
            )
        elif args.input_type == 'image':
            results = search_image(
                image_path=args.input_value,
                video_index=video_index,
                desc_index=desc_index,
                embedding_model=embedding_model,
                caption_processor=caption_processor,
                caption_model=caption_model,
                device=device,
                top_k=args.top_k,
                filter_dict=filter_dict
            )
        elif args.input_type == 'audio':
            results = search_audio(
                audio_path=args.input_value,
                audio_index=audio_index,
                text_index=text_index,
                embedding_model=embedding_model,
                whisper_model=whisper_model,
                device=device,
                top_k=args.top_k,
                filter_dict=filter_dict
            )
        elif args.input_type == 'video':
            results = search_video(
                video_path=args.input_value,
                video_index=video_index,
                desc_index=desc_index,
                embedding_model=embedding_model,
                caption_processor=caption_processor,
                caption_model=caption_model,
                device=device,
                timestamp=args.timestamp,
                top_k=args.top_k,
                filter_dict=filter_dict
            )
        
        # Merge overlapping clips
        if results:
            results = merge_overlapping_clips(results, gap_seconds=3)
        
        # Format and display results
        formatted_results = format_results(results)
        
        print()
        print("=" * 80)
        print(f"SEARCH RESULTS ({len(formatted_results)} found)")
        print("=" * 80)
        
        for i, result in enumerate(formatted_results[:args.top_k], 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Video: {result['video_name']}")
            print(f"   Time: {result['time_range']}")
            print(f"   Source: {result['source']}")
            if result['transcript']:
                print(f"   Transcript: {result['transcript'][:100]}...")
            if result['description']:
                print(f"   Description: {result['description'][:100]}...")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(formatted_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during search: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
