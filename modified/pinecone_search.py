#!/usr/bin/env python3
"""
Pinecone Search Script (Standalone)
Searches Pinecone indexes using CLIP (text/image) and CLAP (audio) embeddings.

Search Pipeline:
    Text Query → CLIP Text Embeddings → Query all indexes
    Image Query → CLIP Image Embeddings (video index) + BLIP Caption → CLIP Text Embeddings (desc index)
    Audio Query → CLAP Audio Embeddings (audio index) + Whisper Transcript → CLIP Text Embeddings (text index)
    Video Query → Extract Frame → Same as Image Query

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
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import torch
from PIL import Image
import cv2

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Try to import optional dependencies
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None
    print("Warning: pydub not available. Install with: pip install pydub")

try:
    import whisper
except ImportError:
    whisper = None
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel,
        ClapProcessor, ClapModel
    )
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    CLIPProcessor = None
    CLIPModel = None
    ClapProcessor = None
    ClapModel = None
    print("Warning: Transformers not available. Install with: pip install transformers")

try:
    from pinecone import Pinecone
except ImportError:
    try:
        import pinecone
        Pinecone = pinecone.Pinecone
    except (ImportError, AttributeError):
        Pinecone = None
        print("Warning: Pinecone not available. Install with: pip install pinecone")


# ============================================================================
# Configuration
# ============================================================================
PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY", ""),
    "video_index_name": os.getenv("PINECONE_FRAME_INDEX", ""),
    "audio_index_name": os.getenv("PINECONE_AUDIO_INDEX", ""),
    "text_index_name": os.getenv("PINECONE_TRANSCRIPT_INDEX", ""),
    "desc_index_name": os.getenv("PINECONE_DESCRIPTION_INDEX", "")
}


# ============================================================================
# Model Loading Functions
# ============================================================================
def load_clip_model(device: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any], str]:
    """Loads the CLIP model and processor."""
    if CLIPProcessor is None or CLIPModel is None:
        return None, None, "cpu"
        
    try:
        print("Loading CLIP model...")
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        model.to(device)
        model.eval()
        print(f"CLIP model loaded successfully on {device}")
        return processor, model, device
    except Exception as e:
        print(f"Could not load CLIP model: {e}")
        return None, None, "cpu"


def load_clap_model(device: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any], str]:
    """Loads the CLAP model and processor."""
    if ClapProcessor is None or ClapModel is None:
        return None, None, "cpu"
        
    try:
        print("Loading CLAP model...")
        model_id = "laion/clap-htsat-unfused"
        processor = ClapProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id)
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        model.to(device)
        model.eval()
        print(f"CLAP model loaded successfully on {device}")
        return processor, model, device
    except Exception as e:
        print(f"Could not load CLAP model: {e}")
        return None, None, "cpu"


def load_captioning_model(device: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any]]:
    """Loads the BLIP image captioning model and processor."""
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        return None, None
    
    try:
        print("Loading BLIP captioning model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        try:
            model.to(device)
        except Exception as e:
            print(f"Warning: Could not move BLIP model to {device}: {e}")
            if device.startswith('cuda'):
                print("Falling back to CPU for BLIP model...")
                device = "cpu"
                model.to(device)
        
        model.eval()
        print(f"BLIP model loaded successfully on {device}")
        return processor, model
    except Exception as e:
        print(f"Could not load captioning model: {e}")
        return None, None


def load_whisper_model(model_name: str = "base"):
    """Loads a Whisper model."""
    if whisper is None:
        return None
    
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Whisper model loaded successfully")
        return model
    except Exception as e:
        print(f"Could not load Whisper model: {e}")
        return None


# ============================================================================
# Embedding Generation Functions
# ============================================================================
def load_audio_pydub(path: str, target_sr: int = 48000) -> Optional[np.ndarray]:
    """Load audio file using pydub and convert to numpy array."""
    if AudioSegment is None:
        return None
        
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0
        elif audio.sample_width == 1:
            samples = (samples.astype(np.float32) - 128) / 128.0
             
        return samples
    except Exception as e:
        print(f"Error loading audio {path}: {e}")
        return None


def get_clip_image_embedding(
    pil_image: Image.Image,
    processor,
    model,
    device: str
) -> Optional[List[float]]:
    """Generates CLIP embedding for a single image."""
    if not processor or not model:
        return None
    
    try:
        inputs = processor(images=[pil_image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting CLIP image embedding: {e}")
        return None


def get_clip_text_embedding(
    text: str,
    processor,
    model,
    device: str
) -> Optional[List[float]]:
    """Generates CLIP embedding for text."""
    if not text or not processor or not model:
        return None
        
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        return features[0].cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting CLIP text embedding: {e}")
        return None


def get_clap_audio_embedding(
    audio_path: str,
    processor,
    model,
    device: str
) -> Optional[List[float]]:
    """Generates CLAP embedding for audio file."""
    if not audio_path or not processor or not model:
        return None
    
    if not os.path.exists(audio_path):
        return None
        
    samples = load_audio_pydub(audio_path, target_sr=48000)
    if samples is None or len(samples) == 0:
        return None
        
    try:
        inputs = processor(audios=[samples], return_tensors="pt", sampling_rate=48000, padding=True).to(device)
        with torch.no_grad():
            features = model.get_audio_features(**inputs)
        return features[0].cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting CLAP audio embedding: {e}")
        return None


def generate_image_caption(
    pil_image: Image.Image, 
    processor, 
    model, 
    device: str
) -> str:
    """Generates text description for an image using BLIP."""
    if not processor or not model:
        return ""
    
    try:
        model_device = next(model.parameters()).device
        inputs = processor(images=[pil_image], return_tensors="pt").to(model_device)
        
        generated_ids = model.generate(**inputs, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_captions[0].strip() if generated_captions else ""
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""


# ============================================================================
# Pinecone Functions
# ============================================================================
def init_pinecone_indexes():
    """Initialize Pinecone indexes."""
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


def query_index(index, query_vec: List[float], top_k: int = 8, filter_dict: Optional[Dict] = None):
    """Queries a Pinecone index with a query vector."""
    if filter_dict:
        return index.query(vector=query_vec, top_k=top_k, include_metadata=True, filter=filter_dict)
    return index.query(vector=query_vec, top_k=top_k, include_metadata=True)


# ============================================================================
# Scoring Functions
# ============================================================================
def compute_final_score(modality_scores: Dict[str, float], threshold: float = 0.75) -> float:
    """
    Computes a final weighted score from multiple modality scores.
    
    Args:
        modality_scores: Dictionary with modality names as keys and scores as values
                         e.g., {'video': 0.33, 'audio': 0.36, 'text': 0.28, 'desc': 0.40}
        threshold: If any raw score >= threshold, use that score directly
        
    Returns:
        Final weighted score (float, rounded to 3 decimal places)
    """
    if not modality_scores:
        return 0.0
    
    max_raw_score = max(modality_scores.values())
    if max_raw_score >= threshold:
        return round(max_raw_score, 3)
    
    weights = {"video": 1.0, "desc": 0.9, "audio": 0.8, "text": 0.6}
    
    present_modalities = [m for m in weights if m in modality_scores]
    if not present_modalities:
        return 0.0
    
    total_w = sum(weights[m] for m in present_modalities)
    base_score = sum(modality_scores[m] * weights[m] for m in present_modalities) / total_w
    
    priority_order = ["video", "desc", "audio", "text"]
    top_mod = next((m for m in priority_order if m in present_modalities), None)
    top_weight = weights[top_mod] if top_mod else 0.6
    
    final_score = base_score * 0.7 + top_weight * 0.3
    return round(final_score, 3)


def compute_audio_search_score(modality_scores: Dict[str, float]) -> float:
    """
    Computes a final score specifically for audio-based searches.
    Priority is given to audio similarity, then to text similarity.
    """
    audio_score = modality_scores.get('audio', 0.0)
    text_score = modality_scores.get('text', 0.0)
    
    if audio_score >= 0.75:
        return round(audio_score, 3)
    
    final_score = (audio_score * 1.0 + text_score * 0.4) / 1.4
    return round(final_score, 3)


# ============================================================================
# Utility Functions
# ============================================================================
def merge_overlapping_clips(search_results: List[Any], gap_seconds: int = 3) -> List[Any]:
    """Merges overlapping or nearby video clips from the same video."""
    if not search_results:
        return []
    
    results_by_video = {}
    for res in search_results:
        if hasattr(res, 'metadata'):
            metadata = res.metadata
        else:
            metadata = res.get('metadata', res)
        
        if isinstance(metadata, dict):
            video_id = metadata.get("video_doc_id") or metadata.get("video_id")
        else:
            video_id = getattr(metadata, 'video_doc_id', None) or getattr(metadata, 'video_id', None)
        
        if video_id not in results_by_video:
            results_by_video[video_id] = []
        results_by_video[video_id].append(res)
    
    final_merged_results = []
    for video_id, clips in results_by_video.items():
        if not clips:
            continue
        
        def get_start_time(clip):
            if hasattr(clip, 'metadata'):
                return clip.metadata.get('start_time', 0)
            return clip.get('metadata', {}).get('start_time', clip.get('start_time', 0))
        
        clips.sort(key=get_start_time)
        
        merged_for_video = []
        current_merge = clips[0]
        
        for next_clip in clips[1:]:
            def get_metadata(clip):
                if hasattr(clip, 'metadata'):
                    return clip.metadata
                return clip.get('metadata', clip)
            
            def get_score(clip):
                if hasattr(clip, 'score'):
                    return clip.score
                return clip.get('score', 0.0)
            
            current_meta = get_metadata(current_merge)
            next_meta = get_metadata(next_clip)
            
            current_end = current_meta.get('end_time', 0) if isinstance(current_meta, dict) else getattr(current_meta, 'end_time', 0)
            next_start = next_meta.get('start_time', 0) if isinstance(next_meta, dict) else getattr(next_meta, 'start_time', 0)
            
            if next_start <= current_end + gap_seconds:
                if isinstance(current_meta, dict):
                    current_meta['end_time'] = max(current_end, next_meta.get('end_time', 0) if isinstance(next_meta, dict) else getattr(next_meta, 'end_time', 0))
                else:
                    current_meta.end_time = max(current_end, next_meta.get('end_time', 0) if isinstance(next_meta, dict) else getattr(next_meta, 'end_time', 0))
                
                current_score = get_score(current_merge)
                next_score = get_score(next_clip)
                if next_score > current_score:
                    if hasattr(current_merge, 'score'):
                        current_merge.score = next_score
                    else:
                        current_merge['score'] = next_score
                
                current_source_str = current_meta.get('source', '') if isinstance(current_meta, dict) else getattr(current_meta, 'source', '')
                next_source_str = next_meta.get('source', '') if isinstance(next_meta, dict) else getattr(next_meta, 'source', '')
                
                current_source = set(str(current_source_str).split(', '))
                next_source = set(str(next_source_str).split(', '))
                current_source.discard('')
                next_source.discard('')
                
                combined_sources = sorted(list(current_source.union(next_source)))
                source_str = ', '.join(combined_sources)
                
                if isinstance(current_meta, dict):
                    current_meta['source'] = source_str
                else:
                    current_meta.source = source_str
            else:
                merged_for_video.append(current_merge)
                current_merge = next_clip
        
        merged_for_video.append(current_merge)
        final_merged_results.extend(merged_for_video)
    
    def get_score_for_sort(clip):
        if hasattr(clip, 'score'):
            return clip.score
        return clip.get('score', 0.0)
    
    final_merged_results.sort(key=get_score_for_sort, reverse=True)
    return final_merged_results


def extract_frame_from_video(video_path: str, timestamp: float = 0.0) -> Image.Image:
    """Extract a single frame from video at given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame from video: {video_path}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# ============================================================================
# Search Functions
# ============================================================================
def search_text(
    query_text: str,
    video_index,
    audio_index,
    text_index,
    desc_index,
    clip_processor,
    clip_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using text query across text and description indexes.
    Uses CLIP text embeddings to query desc index (captions) and text index (transcripts).
    """
    print(f"Generating CLIP text embedding for: '{query_text}'")
    
    query_vector = get_clip_text_embedding(query_text, clip_processor, clip_model, device)
    
    if not query_vector:
        print("Error: Failed to generate text embedding")
        return []
    
    print("Querying text indexes (desc, text)...")
    results_map = {}
    # Only query indexes that contain CLIP text embeddings
    indexes_to_query = {
        'text': text_index,  # Contains CLIP text embeddings from transcripts
        'desc': desc_index   # Contains CLIP text embeddings from BLIP captions
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
    clip_processor,
    clip_model,
    caption_processor,
    caption_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using image query.
    Uses CLIP image embeddings for video index and BLIP caption + CLIP text for desc index.
    """
    print(f"Loading image: {image_path}")
    
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return []
    
    # Generate CLIP image embedding
    print("Generating CLIP image embedding...")
    image_vector = get_clip_image_embedding(pil_image, clip_processor, clip_model, device)
    
    search_tasks = {}
    if image_vector:
        search_tasks['video'] = (video_index, image_vector)
    
    # Generate description and its CLIP text embedding
    if caption_processor and caption_model:
        print("Generating image caption with BLIP...")
        desc = generate_image_caption(pil_image, caption_processor, caption_model, device)
        print(f"Caption: {desc}")
        
        if desc:
            desc_vector = get_clip_text_embedding(desc, clip_processor, clip_model, device)
            if desc_vector:
                search_tasks['desc'] = (desc_index, desc_vector)
    
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
    clap_processor,
    clap_model,
    clip_processor,
    clip_model,
    whisper_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using audio query.
    Uses CLAP audio embeddings for audio index and Whisper transcript + CLIP text for text index.
    """
    print(f"Loading audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return []
    
    search_tasks = {}
    
    # Generate CLAP audio embedding
    print("Generating CLAP audio embedding...")
    audio_vector = get_clap_audio_embedding(audio_path, clap_processor, clap_model, device)
    if audio_vector:
        search_tasks['audio'] = (audio_index, audio_vector)
    
    # Transcribe audio and generate CLIP text embedding
    transcript = None
    if whisper_model:
        try:
            print("Transcribing audio with Whisper...")
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
        text_vector = get_clip_text_embedding(transcript, clip_processor, clip_model, device)
        if text_vector:
            search_tasks['text'] = (text_index, text_vector)
    
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
    clip_processor,
    clip_model,
    caption_processor,
    caption_model,
    device: str,
    timestamp: float = 0.0,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Search using video query (extracts frame and searches).
    Uses CLIP image embeddings for video index and BLIP caption + CLIP text for desc index.
    """
    print(f"Extracting frame from video: {video_path} at {timestamp}s")
    
    try:
        pil_image = extract_frame_from_video(video_path, timestamp)
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return []
    
    # Generate CLIP image embedding
    print("Generating CLIP image embedding from video frame...")
    image_vector = get_clip_image_embedding(pil_image, clip_processor, clip_model, device)
    
    search_tasks = {}
    if image_vector:
        search_tasks['video'] = (video_index, image_vector)
    
    # Generate description and its CLIP text embedding
    if caption_processor and caption_model:
        print("Generating frame caption with BLIP...")
        desc = generate_image_caption(pil_image, caption_processor, caption_model, device)
        print(f"Caption: {desc}")
        
        if desc:
            desc_vector = get_clip_text_embedding(desc, clip_processor, clip_model, device)
            if desc_vector:
                search_tasks['desc'] = (desc_index, desc_vector)
    
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


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Search Pinecone indexes using CLIP and CLAP embeddings',
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
    print("PINECONE SEARCH (CLIP + CLAP)")
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
        if Pinecone:
            pc = Pinecone(api_key=PINECONE_CONFIG['api_key'])
            print(f"  {pc.list_indexes().names()}")
        return 1
    
    print("✓ Indexes initialized")
    print()
    
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load CLIP model (needed for all search types)
    clip_processor, clip_model, device = load_clip_model(device)
    if not clip_model:
        print("Error: Failed to load CLIP model")
        return 1
    print(f"✓ CLIP loaded on {device}")
    
    # Load additional models based on search type
    caption_processor = None
    caption_model = None
    clap_processor = None
    clap_model = None
    whisper_model = None
    
    if args.input_type in ['image', 'video']:
        print("Loading BLIP captioning model...")
        caption_processor, caption_model = load_captioning_model(device)
        if caption_processor and caption_model:
            print("✓ BLIP loaded")
    
    if args.input_type == 'audio':
        print("Loading CLAP model...")
        clap_processor, clap_model, _ = load_clap_model(device)
        if clap_processor and clap_model:
            print("✓ CLAP loaded")
        
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
                clip_processor=clip_processor,
                clip_model=clip_model,
                device=device,
                top_k=args.top_k,
                filter_dict=filter_dict
            )
        elif args.input_type == 'image':
            results = search_image(
                image_path=args.input_value,
                video_index=video_index,
                desc_index=desc_index,
                clip_processor=clip_processor,
                clip_model=clip_model,
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
                clap_processor=clap_processor,
                clap_model=clap_model,
                clip_processor=clip_processor,
                clip_model=clip_model,
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
                clip_processor=clip_processor,
                clip_model=clip_model,
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
