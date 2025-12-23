#!/usr/bin/env python3
"""
Create Embeddings Script (Standalone)
Generates embeddings for videos using CLIP (text/image) and CLAP (audio) and uploads to Pinecone.

Embedding Pipeline:
    Video Input
    ├── Frame Path
    │   ├── Extract Key Frames → BLIP Captioning → CLIP Text Embeddings → Pinecone (clip-text/captions)
    │   └── Extract Key Frames → CLIP Image Embeddings → Pinecone (clip-image/frames)
    └── Audio Path
        ├── Audio Segments → CLAP Audio Embeddings → Pinecone (clap-audio)
        └── Audio Segments → Whisper Transcription → Transcript Chunks → CLIP Text Embeddings → Pinecone (clip-transcript)

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
import json
import queue
import threading
import tempfile
import subprocess
import concurrent.futures
import warnings
from pathlib import Path
from time import sleep
from contextlib import redirect_stderr
from typing import List, Dict, Tuple, Optional, Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from pydub import AudioSegment

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
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import requests
except ImportError:
    requests = None
    print("Warning: requests not available. Install with: pip install requests")

try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
except ImportError:
    SceneManager = None
    print("Warning: scenedetect not available. Install with: pip install scenedetect")

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

BATCH_SIZE = 16  # Number of scenes to process in each batch


# ============================================================================
# Model Loading Functions
# ============================================================================
def load_clip_model(device: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any]]:
    """Loads the CLIP model and processor."""
    if CLIPProcessor is None or CLIPModel is None:
        return None, None
        
    try:
        print("Loading CLIP model...")
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        model.to(device)
        print(f"CLIP model loaded successfully on {device}")
        return processor, model
    except Exception as e:
        print(f"Could not load CLIP model: {e}")
        return None, None


def load_clap_model(device: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any]]:
    """Loads the CLAP model and processor."""
    if ClapProcessor is None or ClapModel is None:
        return None, None
        
    try:
        print("Loading CLAP model...")
        model_id = "laion/clap-htsat-unfused"
        processor = ClapProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id)
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        model.to(device)
        print(f"CLAP model loaded successfully on {device}")
        return processor, model
    except Exception as e:
        print(f"Could not load CLAP model: {e}")
        return None, None


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
        
        print(f"BLIP model loaded successfully on {device}")
        return processor, model
    except Exception as e:
        print(f"Could not load captioning model: {e}")
        return None, None


# ============================================================================
# Embedding Generation Functions
# ============================================================================
def load_audio_pydub(path: str, target_sr: int = 48000) -> Optional[np.ndarray]:
    """Load audio file using pydub and convert to numpy array."""
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


def get_clip_image_embeddings(
    pil_images: List[Image.Image],
    processor,
    model,
    device: str
) -> torch.Tensor:
    """Generates CLIP embeddings for images."""
    if not pil_images or not processor or not model:
        return torch.empty((0,))
    
    try:
        inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features
    except Exception as e:
        print(f"Error getting CLIP image embeddings: {e}")
        return torch.empty((0,))


def get_clip_text_embeddings(
    texts: List[str],
    processor,
    model,
    device: str
) -> torch.Tensor:
    """Generates CLIP embeddings for texts."""
    if not texts or not processor or not model:
        return torch.empty((0,))
        
    try:
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        return features
    except Exception as e:
        print(f"Error getting CLIP text embeddings: {e}")
        return torch.empty((0,))


def get_clap_audio_embeddings(
    audio_paths: List[str],
    processor,
    model,
    device: str
) -> torch.Tensor:
    """Generates CLAP embeddings for audio files."""
    if not audio_paths or not processor or not model:
        return torch.empty((0,))
        
    audios = []
    
    for path in audio_paths:
        if path and os.path.exists(path):
            samples = load_audio_pydub(path, target_sr=48000)
            if samples is not None and len(samples) > 0:
                audios.append(samples)
            else:
                audios.append(np.zeros(48000, dtype=np.float32))
        else:
            audios.append(np.zeros(48000, dtype=np.float32))
    
    if not audios:
        return torch.empty((0,))
        
    try:
        inputs = processor(audios=audios, return_tensors="pt", sampling_rate=48000, padding=True).to(device)
        with torch.no_grad():
            features = model.get_audio_features(**inputs)
        return features
    except Exception as e:
        print(f"Error getting CLAP audio embeddings: {e}")
        return torch.empty((0,))


def generate_scene_descriptions(
    pil_images: List[Image.Image], 
    processor, 
    model, 
    device: str
) -> List[str]:
    """Generates text descriptions for a batch of image frames using BLIP."""
    if not processor or not model or not pil_images:
        return [""] * len(pil_images)
    
    try:
        model_device = next(model.parameters()).device
        inputs = processor(images=pil_images, return_tensors="pt").to(model_device)
        
        generated_ids = model.generate(**inputs, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [caption.strip() for caption in generated_captions]
    except Exception as e:
        print(f"Error during batch caption generation: {e}")
        return [""] * len(pil_images)


# ============================================================================
# Video Processing Functions
# ============================================================================
def extract_audio_from_video(video_path: str) -> Optional[str]:
    """Extracts audio from a video and saves it as a WAV file."""
    if not os.path.exists(video_path):
        return None

    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = temp_audio_file.name
    temp_audio_file.close()
    
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", "-f", "wav", "-y", audio_path, "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 44:
            return audio_path
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"Error extracting audio from video: {e}", file=sys.stderr)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return None


def find_segments_in_scene_range(scene_start_time: float, scene_end_time: float, segments: List[Dict], max_length: int = 10000) -> str:
    """Finds all transcript segments that overlap with the scene's time range."""
    if not segments:
        return ''
    
    overlapping_segments = []
    
    for segment in segments:
        segment_start = segment.get('start', 0)
        segment_end = segment.get('end', 0)
        segment_text = segment.get('text', '').strip()
        
        if segment_start < scene_end_time and segment_end > scene_start_time and segment_text:
            overlapping_segments.append((segment_start, segment_text))
    
    overlapping_segments.sort(key=lambda x: x[0])
    transcript_text = ' '.join(text for _, text in overlapping_segments)
    
    if len(transcript_text) > max_length:
        transcript_text = transcript_text[:max_length].rsplit(' ', 1)[0] + '...'
    
    return transcript_text.strip()


def detect_scenes_with_adaptive_detector(video_path: str) -> List:
    """Detect scenes using AdaptiveDetector."""
    if SceneManager is None:
        return []
        
    video_stream = None
    with open(os.devnull, 'w') as devnull:
        with redirect_stderr(devnull):
            try:
                video_stream = open_video(video_path)
                scene_manager = SceneManager()
                custom_weights = ContentDetector.Components(
                    delta_hue=1.0,
                    delta_sat=1.0,
                    delta_lum=5.0,
                    delta_edges=0.0
                )
                scene_manager.add_detector(AdaptiveDetector(
                    adaptive_threshold=1.0,
                    weights=custom_weights
                ))
                scene_manager.detect_scenes(video_stream, show_progress=False)
                scene_list = scene_manager.get_scene_list()
                return scene_list
            except Exception as e:
                print(f"Scene detection failed for {os.path.basename(video_path)}: {e}")
                return []
            finally:
                if video_stream:
                    try:
                        if hasattr(video_stream, 'close'):
                            video_stream.close()
                    except Exception:
                        pass


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    if not os.path.exists(video_path):
        return 0.0
    
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            if duration > 0:
                return duration
    except (FileNotFoundError, Exception):
        pass
    
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                if duration > 0:
                    return duration
    except Exception:
        pass
    
    return 0.0


def split_video_into_chunks(video_path: str, chunk_duration_seconds: float = 600.0) -> List[Tuple[float, float, str]]:
    """Split video into chunks of specified duration."""
    duration = get_video_duration(video_path)
    if duration == 0.0 or duration <= chunk_duration_seconds:
        return [(0.0, duration, video_path)]
    
    chunks = []
    chunk_dir = tempfile.mkdtemp(prefix="video_chunks_")
    start_time = 0.0
    
    for i in range(int(duration // chunk_duration_seconds) + 1):
        end_time = min(start_time + chunk_duration_seconds, duration)
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.mp4")
        
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-i", video_path,
            "-t", str(end_time - start_time), "-c", "copy", 
            "-avoid_negative_ts", "make_zero", "-y", chunk_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                cmd[9] = "libx264"
                subprocess.run(cmd, capture_output=True, text=True, check=False)

            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                chunks.append((start_time, end_time, chunk_path))
        except Exception:
            pass
        start_time = end_time

    return chunks if chunks else [(0.0, duration, video_path)]


def resize_frame_optimized(frame_np: np.ndarray, max_size: int = 512) -> Image.Image:
    """Resize frame to reduce processing time while maintaining aspect ratio."""
    pil_image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    return pil_image


def prepare_scene_data(args: Tuple) -> Optional[Dict]:
    """Prepares all necessary data for a single scene for batch processing."""
    chunk_path, scene, fps, chunk_start_offset, video_name, video_id, global_scene_index, transcript_segments, full_audio_segment = args
    
    abs_start_time = chunk_start_offset + scene[0].get_seconds()
    abs_end_time = chunk_start_offset + scene[1].get_seconds()

    start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
    middle_frame = start_frame + (end_frame - start_frame) // 2
    
    cap = cv2.VideoCapture(chunk_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame_np = cap.read()
    cap.release()
    if not ret:
        return None
    
    scene_audio_path = ""
    if full_audio_segment:
        try:
            start_ms = int(abs_start_time * 1000)
            end_ms = int(abs_end_time * 1000)
            scene_audio_clip = full_audio_segment[start_ms:end_ms]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_clip:
                scene_audio_path = temp_clip.name
                scene_audio_clip.export(scene_audio_path, format="wav")
        except Exception:
            scene_audio_path = ""
    
    transcript_text = ''
    if transcript_segments:
        transcript_text = find_segments_in_scene_range(abs_start_time, abs_end_time, transcript_segments)
    
    return {
        "pil_image": resize_frame_optimized(frame_np),
        "audio_path": scene_audio_path,
        "transcript": transcript_text,
        "video_name": video_name,
        "video_id": video_id,
        "scene_global_index": global_scene_index,
        "abs_start_time": abs_start_time,
        "abs_end_time": abs_end_time,
    }


# ============================================================================
# Download Functions
# ============================================================================
def download_from_url(url: str, output_path: Optional[str] = None, chunk_size: int = 8 * 1024 * 1024, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Downloads a file from a URL."""
    if requests is None:
        print("Error: requests library not available")
        return None
        
    if progress_callback is None:
        progress_callback = print
    
    if url.startswith('file://'):
        local_path = url[7:]
    elif url.startswith('/') and os.path.exists(url):
        local_path = url
    else:
        local_path = None
    
    if local_path:
        if os.path.exists(local_path):
            progress_callback(f"Using local file: {local_path}")
            return local_path
        else:
            raise Exception(f"Local file path does not exist: {local_path}")
    
    try:
        if output_path is None:
            file_ext = os.path.splitext(url.split('?')[0])[1] or '.tmp'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            output_path = temp_file.name
            temp_file.close()
        
        progress_callback("Downloading...")
        
        session = requests.Session()
        response = session.get(url, stream=True, timeout=(30, 300), allow_redirects=True)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.reason}")
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            progress_callback(f"File size: {total_size / (1024*1024):.2f} MB")
        
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        progress_callback(f"Download complete: {downloaded / (1024*1024):.2f} MB")
        session.close()
        return output_path
        
    except Exception as e:
        progress_callback(f"Error downloading: {e}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None


def download_transcript_from_url(url: str) -> Optional[list]:
    """Downloads and parses a transcript JSON from a URL."""
    if url.startswith('file://'):
        local_path = url[7:]
    elif url.startswith('/') and os.path.exists(url):
        local_path = url
    else:
        local_path = None
    
    if local_path:
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            if isinstance(transcript_data, list):
                return transcript_data
            elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
                return transcript_data['segments']
            elif isinstance(transcript_data, dict) and 'transcript' in transcript_data:
                return transcript_data['transcript']
            else:
                return transcript_data if isinstance(transcript_data, list) else None
        except Exception as e:
            print(f"Error reading local transcript: {e}")
            return None
    
    if requests is None:
        print("Error: requests library not available")
        return None
        
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        transcript_data = response.json()
        
        if isinstance(transcript_data, list):
            return transcript_data
        elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
            return transcript_data['segments']
        elif isinstance(transcript_data, dict) and 'transcript' in transcript_data:
            return transcript_data['transcript']
        else:
            return transcript_data if isinstance(transcript_data, list) else None
            
    except Exception as e:
        print(f"Error downloading transcript: {e}")
        return None


# ============================================================================
# Pinecone Upload Manager
# ============================================================================
class PineconeUploadManager:
    """Manages background uploads to Pinecone using a queue and worker threads."""
    
    def __init__(self, index, num_workers: int = 4, batch_size: int = 100, log_callback: Optional[Callable[[str], None]] = None):
        self.index = index
        self.batch_size = batch_size
        self.upload_queue = queue.Queue()
        self.workers = []
        self.is_running = True
        self.uploaded_count = 0
        self.errors = []
        self.lock = threading.Lock()
        self.log_callback = log_callback or print
        self.index_name = getattr(index, 'name', 'unknown')
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._upload_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

    def _upload_worker(self, worker_id: int):
        while self.is_running or not self.upload_queue.empty():
            try:
                vectors_batch = self.upload_queue.get(timeout=1)
                try:
                    if vectors_batch:
                        batch_size = len(vectors_batch)
                        max_retries = 3
                        retry_count = 0
                        success = False
                        
                        while retry_count < max_retries and not success:
                            try:
                                self.index.upsert(vectors=vectors_batch, namespace="__default__")
                                with self.lock:
                                    self.uploaded_count += len(vectors_batch)
                                success = True
                            except Exception as e:
                                error_str = str(e).lower()
                                if any(keyword in error_str for keyword in ['network', 'connection', 'timeout']) and retry_count < max_retries - 1:
                                    retry_count += 1
                                    sleep(2.0 * retry_count)
                                    continue
                                with self.lock:
                                    self.errors.append(f"Worker {worker_id}: {e}")
                                break
                finally:
                    self.upload_queue.task_done()
            except queue.Empty:
                continue

    def add_to_queue(self, vectors: List):
        """Add vectors to upload queue."""
        if vectors:
            self.upload_queue.put(vectors)

    def wait_for_completion(self):
        """Wait for all queued uploads to complete."""
        self.upload_queue.join()
        self.log_callback(f"  ✓ [{self.index_name}] All uploads completed. Total: {self.uploaded_count}")

    def stop(self):
        """Stop the upload workers."""
        self.is_running = False
        for worker in self.workers:
            worker.join(timeout=2)
    
    def get_stats(self):
        """Get upload statistics."""
        with self.lock:
            return {"uploaded": self.uploaded_count, "errors": len(self.errors)}


# ============================================================================
# Pinecone Initialization
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


# ============================================================================
# Batch Processing
# ============================================================================
def process_batch(
    batch_data: List[Dict[str, Any]],
    clip_processor,
    clip_model,
    clap_processor,
    clap_model,
    caption_processor,
    caption_model,
    device: str
) -> List[Dict[str, Any]]:
    """Processes a batch of scene data to generate embeddings using CLIP, CLAP, and BLIP."""
    results = []
    pil_images = [item['pil_image'] for item in batch_data]
    audio_paths = [item['audio_path'] for item in batch_data]
    transcripts = [item['transcript'] for item in batch_data]
    
    transcripts_for_embedding = [t if t and t.strip() else "" for t in transcripts]

    # BLIP: Generate descriptions (captions)
    descriptions = generate_scene_descriptions(pil_images, caption_processor, caption_model, device)

    # 1. Video Frames → CLIP Image Embeddings → Pinecone (clip-image/frames)
    image_embeddings = get_clip_image_embeddings(
        pil_images=pil_images,
        processor=clip_processor,
        model=clip_model,
        device=device
    )
    
    # 2. Audio Segments → CLAP Audio Embeddings → Pinecone (clap-audio)
    audio_embeddings = get_clap_audio_embeddings(
        audio_paths=audio_paths,
        processor=clap_processor,
        model=clap_model,
        device=device
    )
    
    # 3. Transcript Chunks → CLIP Text Embeddings → Pinecone (clip-transcript)
    text_embeddings = get_clip_text_embeddings(
        texts=transcripts_for_embedding,
        processor=clip_processor,
        model=clip_model,
        device=device
    )
    
    # 4. BLIP Captions → CLIP Text Embeddings → Pinecone (clip-text/captions)
    desc_embeddings = get_clip_text_embeddings(
        texts=descriptions,
        processor=clip_processor,
        model=clip_model,
        device=device
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
            if transcripts_for_embedding[i]:
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


def process_video_chunk(args: Tuple) -> List[str]:
    """Processes a single video chunk."""
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
        clip_processor,
        clip_model,
        clap_processor,
        clap_model,
        caption_processor,
        caption_model,
        device,
        log_callback
    ) = args

    chunk_name = os.path.basename(chunk_path)
    log_callback(f"\n{'='*80}")
    log_callback(f"📹 Processing Chunk: {chunk_name} (starts at {chunk_start_offset:.2f}s)")
    log_callback(f"{'='*80}")

    log_callback(f"  🔍 Detecting scenes in chunk...")
    scene_list = detect_scenes_with_adaptive_detector(chunk_path)
    if not scene_list:
        log_callback(f"  ⚠ No scenes detected in chunk {chunk_name}")
        return []
    
    log_callback(f"  ✓ Detected {len(scene_list)} scenes in chunk")

    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    log_callback(f"  📦 Preparing scene data...")
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
    chunk_scene_data.sort(key=lambda x: x['scene_global_index'])

    num_batches = (len(chunk_scene_data) + BATCH_SIZE - 1) // BATCH_SIZE
    log_callback(f"  🧠 Generating embeddings in {num_batches} batch(es)...")
    
    IMMEDIATE_BATCH_SIZE = 20
    video_vectors_batch = []
    audio_vectors_batch = []
    text_vectors_batch = []
    desc_vectors_batch = []
    
    for i in range(0, len(chunk_scene_data), BATCH_SIZE):
        batch = chunk_scene_data[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        log_callback(f"    📊 Batch {batch_num}/{num_batches}: Processing {len(batch)} scenes...")
        
        batch_results = process_batch(
            batch, 
            clip_processor, clip_model, 
            clap_processor, clap_model,
            caption_processor, caption_model, 
            device
        )

        for result in batch_results:
            if result['video_vector']:
                video_vectors_batch.append(result['video_vector'])
                if len(video_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    video_upload_manager.add_to_queue(video_vectors_batch)
                    video_vectors_batch = []
            
            if result['audio_vector']:
                audio_vectors_batch.append(result['audio_vector'])
                if len(audio_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    audio_upload_manager.add_to_queue(audio_vectors_batch)
                    audio_vectors_batch = []
            
            if result['text_vector']:
                text_vectors_batch.append(result['text_vector'])
                if len(text_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    text_upload_manager.add_to_queue(text_vectors_batch)
                    text_vectors_batch = []
            
            if result['desc_vector']:
                desc_vectors_batch.append(result['desc_vector'])
                if len(desc_vectors_batch) >= IMMEDIATE_BATCH_SIZE:
                    desc_upload_manager.add_to_queue(desc_vectors_batch)
                    desc_vectors_batch = []
        
        log_callback(f"    ✓ Batch {batch_num}/{num_batches}: Processed {len(batch)} scenes")
    
    # Upload remaining vectors
    if video_vectors_batch:
        video_upload_manager.add_to_queue(video_vectors_batch)
    if audio_vectors_batch:
        audio_upload_manager.add_to_queue(audio_vectors_batch)
    if text_vectors_batch:
        text_upload_manager.add_to_queue(text_vectors_batch)
    if desc_vectors_batch:
        desc_upload_manager.add_to_queue(desc_vectors_batch)
    
    log_callback(f"  ✅ Chunk {chunk_name} processing complete")
    log_callback(f"{'='*80}\n")

    return temp_audio_clips


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Create embeddings for videos using CLIP + CLAP and upload to Pinecone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using local files:
  python create_embeddings.py --video video.mp4 --transcript transcript.json --name "my_video"
  
  # Using URLs:
  python create_embeddings.py --video-url "https://..." --transcript-url "https://..." --name "my_video"
  
  # With custom video ID:
  python create_embeddings.py --video video.mp4 --transcript transcript.json --name "my_video" --video-id "custom-123"
        """
    )
    
    parser.add_argument('--video', type=str, default=None, help='Path to local video file')
    parser.add_argument('--video-url', type=str, default=None, help='URL to video file')
    parser.add_argument('--transcript', type=str, default=None, help='Path to local transcript JSON file')
    parser.add_argument('--transcript-url', type=str, default=None, help='URL to transcript JSON file')
    parser.add_argument('--name', '--video-name', dest='video_name', type=str, required=True, help='Name/identifier for the video')
    parser.add_argument('--video-id', type=str, default=None, help='Unique video ID (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    if not args.video and not args.video_url:
        print("Error: Either --video or --video-url must be provided")
        return 1
    
    if not args.transcript and not args.transcript_url:
        print("Error: Either --transcript or --transcript-url must be provided")
        return 1
    
    if args.video and not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if args.transcript and not os.path.exists(args.transcript):
        print(f"Error: Transcript file not found: {args.transcript}")
        return 1
    
    if not PINECONE_CONFIG['api_key']:
        print("Error: PINECONE_API_KEY not found in .env file")
        return 1
    
    video_id = args.video_id or str(uuid.uuid4())
    
    print("=" * 80)
    print("CREATE EMBEDDINGS (CLIP + CLAP)")
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
    
    try:
        # Initialize Pinecone
        print("Initializing Pinecone indexes...")
        video_index, audio_index, text_index, desc_index = init_pinecone_indexes()
        
        if not all([video_index, audio_index, text_index, desc_index]):
            print("Error: Failed to initialize Pinecone indexes")
            return 1
        
        print("✓ Indexes initialized")
        
        # Get video path
        video_path = args.video
        if args.video_url:
            print("Downloading video from URL...")
            video_path = download_from_url(args.video_url, progress_callback=print)
            if not video_path or not os.path.exists(video_path):
                print(f"Error: Failed to download video from URL")
                return 1
            print(f"✓ Video downloaded to: {video_path}")
        
        # Load transcript
        transcript_segments = None
        if args.transcript:
            print(f"Loading transcript from local file: {args.transcript}")
            with open(args.transcript, 'r') as f:
                transcript_data = json.load(f)
                if isinstance(transcript_data, list):
                    transcript_segments = transcript_data
                elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
                    transcript_segments = transcript_data['segments']
            print(f"✓ Loaded {len(transcript_segments) if transcript_segments else 0} transcript segments")
        elif args.transcript_url:
            print(f"Downloading transcript from URL...")
            transcript_segments = download_transcript_from_url(args.transcript_url)
            if transcript_segments:
                print(f"✓ Loaded {len(transcript_segments)} transcript segments from URL")
        
        # Load models
        print("\nLoading models...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device.startswith('cuda'):
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            except:
                pass
        print()
        
        clip_processor, clip_model = load_clip_model(device=device)
        if not clip_model:
            print("Error: Failed to load CLIP model")
            return 1
            
        clap_processor, clap_model = load_clap_model(device=device)
        if not clap_model:
            print("Warning: Failed to load CLAP model")
            
        caption_processor, caption_model = load_captioning_model(device=device)
        
        print("=" * 80 + "\n")
        
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
        
        # Pre-detect scenes
        scene_counts = []
        for i, (start, end, chunk_path) in enumerate(chunks):
            count = len(detect_scenes_with_adaptive_detector(chunk_path))
            scene_counts.append(count)
        
        total_scenes = sum(scene_counts)
        print(f"✓ Total scenes detected: {total_scenes}")
        
        # Process chunks
        print(f"\n🚀 Processing {len(chunks)} chunk(s)...")
        chunk_scene_offset = 0
        temp_audio_clips = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            chunk_args_list = []
            
            for i, (chunk_start, chunk_end, chunk_path) in enumerate(chunks):
                chunk_args = (
                    chunk_path, chunk_start, args.video_name, video_id, chunk_scene_offset,
                    transcript_segments, full_audio_segment,
                    video_upload_manager, audio_upload_manager, text_upload_manager, desc_upload_manager,
                    clip_processor, clip_model, clap_processor, clap_model, caption_processor, caption_model,
                    device,
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
        print(f"Video embeddings (CLIP image):     {video_stats['uploaded']} ({video_stats['errors']} errors)")
        print(f"Audio embeddings (CLAP):           {audio_stats['uploaded']} ({audio_stats['errors']} errors)")
        print(f"Text embeddings (CLIP text):       {text_stats['uploaded']} ({text_stats['errors']} errors)")
        print(f"Description embeddings (CLIP text): {desc_stats['uploaded']} ({desc_stats['errors']} errors)")
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


if __name__ == "__main__":
    sys.exit(main())
