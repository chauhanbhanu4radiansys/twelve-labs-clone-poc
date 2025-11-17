import streamlit as st
import boto3
from botocore.config import Config as BotoConfig
from pymongo import MongoClient
from bson import ObjectId
import os
import logging

# Configure logging to suppress ImageBind warnings that can cause issues in multi-threaded environments
# This prevents "I/O operation on closed file" errors when stderr is redirected in parallel threads
# We need to configure this BEFORE importing imagebind to prevent logging issues

# Create a null handler that discards all log messages
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Configure ImageBind loggers to use null handler (thread-safe)
try:
    # Completely disable logging for imagebind modules to prevent stderr issues in threads
    logging.getLogger('imagebind.data').disabled = True
    logging.getLogger('imagebind').disabled = True
    
    # Also set level to CRITICAL and add null handler as backup
    for logger_name in ['imagebind', 'imagebind.data']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers = []  # Clear existing handlers
        logger.addHandler(NullHandler())
        logger.propagate = False  # Don't propagate to root logger
except Exception:
    pass  # If logging configuration fails, continue anyway

# Suppress deprecation warnings from dependencies
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='imagebind')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*torch.utils._pytree._register_pytree_node is deprecated.*')
warnings.filterwarnings('ignore', message='.*Torchaudio.*backend.*')

# Pinecone import with version handling
try:
    from pinecone import Pinecone
except ImportError:
    try:
        import pinecone
        Pinecone = pinecone.Pinecone
    except (ImportError, AttributeError):
        st.error("Failed to import Pinecone. Please ensure 'pinecone' package is installed: pip install pinecone")
        st.stop()
try:
    import openai
except ImportError:
    st.error("Failed to import OpenAI. Please install it with: pip install openai")
    st.stop()
try:
    import whisper
except ImportError:
    st.error("Failed to import OpenAI Whisper. Please install it with: pip install openai-whisper")
    st.stop()
try:
    import torchaudio
except ImportError:
    st.error("Failed to import torchaudio. Please ensure it is installed.")
    st.stop()

import tempfile
import time
import uuid
from datetime import datetime
import cv2
from PIL import Image
import torch
import sys

# Store original warning function at module level to avoid recursion
# _original_logging_warning = logging.warning

# Patch ImageBind's logging after importing to prevent stderr issues
# This must be done after importing imagebind modules
# def _patch_imagebind_logging():
#     """Monkey-patch ImageBind's logging to prevent stderr issues in multi-threaded environments."""
#     try:
#         import imagebind.data as imagebind_data_module
        
#         def safe_warning(msg, *args, **kwargs):
#             """Safe warning that checks if stderr is available before logging."""
#             try:
#                 # Check if stderr is available and not closed
#                 if sys.stderr and (not hasattr(sys.stderr, 'closed') or not sys.stderr.closed):
#                     # Try to write a test to see if it's actually writable
#                     try:
#                         sys.stderr.write('')
#                         sys.stderr.flush()
#                         _original_logging_warning(msg, *args, **kwargs)
#                     except (ValueError, OSError, AttributeError):
#                         # stderr is closed or not writable, silently ignore
#                         pass
#             except (ValueError, AttributeError, OSError):
#                 # Silently ignore if stderr is closed or unavailable
#                 pass
        
#         # Patch logging.warning globally to be safe
#         logging.warning = safe_warning
        
#         # Also patch in imagebind.data module if it has its own logging reference
#         if hasattr(imagebind_data_module, 'logging'):
#             imagebind_data_module.logging.warning = safe_warning
#     except Exception:
#         pass  # If patching fails, continue anyway

# Import ImageBind modules
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# Apply the patch after import
# _patch_imagebind_logging()
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector
import subprocess
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from pymongo.errors import ConnectionFailure, OperationFailure
import queue
import threading
import concurrent.futures
from io import BytesIO
import numpy as np
from contextlib import redirect_stderr
from typing import Any, Callable, Dict, List, Tuple
import json
import re
from functools import wraps
from time import sleep
from transformers import BlipProcessor, BlipForConditionalGeneration
from pydub import AudioSegment

# --- Constants ---
BATCH_SIZE = 16


# --- Helper Functions ---
def retry_on_network_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry functions on network errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if it's a network-related error
                    if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'axios', 'http', 'request']):
                        last_exception = e
                        retries += 1
                        if retries < max_retries:
                            sleep(current_delay)
                            current_delay *= backoff
                            continue
                    # If it's not a network error, raise immediately
                    raise
            # If we exhausted retries, raise the last exception
            raise last_exception
        return wrapper
    return decorator

def time_str_to_seconds(time_str: str) -> int:
    """Converts HH:MM:SS or MM:SS to seconds."""
    parts = time_str.split(':')
    seconds = 0
    try:
        if len(parts) == 3:  # HH:MM:SS
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
            seconds = int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0  # Return 0 if conversion fails
    return seconds


def format_duration(seconds: float | None) -> str:
    """Formats a duration in seconds into a human-readable string (e.g., '1m 23s')."""
    if seconds is None:
        return "N/A"
    try:
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    except (ValueError, TypeError):
        return "N/A"


# --- Page and App Configuration ---
st.set_page_config(
    page_title="Video Prism",
    page_icon="🎬",
    layout="wide"
)

# --- Configuration Loading from Streamlit Secrets ---
def load_config():
    """Loads configuration from Streamlit's secrets."""
    config = {}
    try:
        # MinIO Config
        config["minio"] = {
            "endpoint": st.secrets["minio"]["endpoint"],
            "access_key": st.secrets["minio"]["access_key"],
            "secret_key": st.secrets["minio"]["secret_key"],
            "bucket_name": st.secrets["minio"]["bucket_name"],
            "transcripts_bucket_name": st.secrets["minio"].get("transcripts_bucket_name", "transcripts"),
            "secure": st.secrets["minio"].get("secure", False)
        }
        # MongoDB Config
        config["mongodb"] = {
            "uri": st.secrets["mongodb"]["uri"],
            "database_name": st.secrets["mongodb"]["database_name"],
            "collection_name": st.secrets["mongodb"]["collection_name"],
        }
        # Pinecone Config
        config["pinecone"] = {
            "api_key": st.secrets["pinecone"]["api_key"],
            "video_index_name": "video-search",
            "audio_index_name": "audio-search",
            "text_index_name": "text-search",
            "desc_index_name": "desc-search"
        }
        # OpenAI Config
        try:
            if "openai" in st.secrets:
                if "api_key" in st.secrets["openai"]:
                    config["openai"] = {
                        "api_key": st.secrets["openai"]["api_key"]
                    }
                else:
                    print("Warning: OpenAI section found but 'api_key' key is missing")
                    config["openai"] = None
            else:
                print("Warning: 'openai' section not found in st.secrets")
                config["openai"] = None
        except Exception as e:
            print(f"Warning: Could not load OpenAI config: {e}")
            import traceback
            traceback.print_exc()
            config["openai"] = None

        return config
    except KeyError as e:
        st.error(f"Configuration Error: Missing secret '{e.args[0]}'. Please check your secrets.toml file.")
        return None

config = load_config()

# --- Client Initialization (Cached) ---

@st.cache_resource
def init_minio_client(config):
    """Initializes and caches the MinIO (S3) client."""
    if not config:
        return None
    try:
        @retry_on_network_error(max_retries=3, delay=2.0)
        def create_s3_client():
            return boto3.client(
                "s3",
                endpoint_url=f"http{'s' if config['minio']['secure'] else ''}://{config['minio']['endpoint']}",
                aws_access_key_id=config["minio"]["access_key"],
                aws_secret_access_key=config["minio"]["secret_key"],
                config=BotoConfig(
                    connect_timeout=10,
                    read_timeout=30,
                    retries={'max_attempts': 3}
                )
            )
        
        s3_client = create_s3_client()
        
        # Check bucket existence and create if necessary with retry
        @retry_on_network_error(max_retries=3, delay=1.0)
        def check_and_create_bucket(bucket_name):
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    s3_client.create_bucket(Bucket=bucket_name)
                    st.toast(f"MinIO bucket '{bucket_name}' created.", icon="💾")
                else:
                    raise
        
        check_and_create_bucket(config["minio"]["bucket_name"])
        check_and_create_bucket(config["minio"]["transcripts_bucket_name"])
        return s3_client
    except (NoCredentialsError, PartialCredentialsError):
        st.error("MinIO connection failed: AWS credentials not found.")
        return None
    except ClientError as e:
        error_msg = str(e)
        if "network" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            st.error(f"MinIO connection failed due to network error: {e}. Please check your connection and try again.")
        else:
            st.error(f"MinIO connection failed: {e}")
        return None
    except Exception as e:
        error_msg = str(e)
        if "network" in error_msg.lower() or "connection" in error_msg.lower():
            st.error(f"MinIO connection failed due to network error: {e}. Please check your connection and try again.")
        else:
            st.error(f"MinIO connection failed: {e}")
        return None

@st.cache_resource
def init_mongo_collection(config):
    """Initializes and caches the MongoDB client and returns the collection."""
    if not config:
        return None
    try:
        client = MongoClient(config["mongodb"]["uri"], serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Test connection
        db = client[config["mongodb"]["database_name"]]
        videos_collection = db[config["mongodb"]["collection_name"]]
        return videos_collection
    except ConnectionFailure:
        st.error("MongoDB connection failed. Check your URI and network access.")
        return None
    except OperationFailure as e:
        st.error(f"MongoDB authentication failed: {e.details.get('errmsg', '')}")
        return None

@st.cache_resource
def init_pinecone_indexes(config):
    """Initializes and caches the Pinecone indexes."""
    if not config:
        return None, None, None, None
    try:
        # Add timeout and retry logic for Pinecone initialization
        @retry_on_network_error(max_retries=3, delay=2.0)
        def init_pinecone_client():
            return Pinecone(api_key=config["pinecone"]["api_key"])
        
        pc = init_pinecone_client()
        video_index_name = config["pinecone"]["video_index_name"]
        audio_index_name = config["pinecone"]["audio_index_name"]
        text_index_name = config["pinecone"]["text_index_name"]
        desc_index_name = config["pinecone"]["desc_index_name"]

        @retry_on_network_error(max_retries=3, delay=1.0)
        def connect_to_index(index_name):
            try:
                index_list = pc.list_indexes().names()
                if index_name not in index_list:
                    st.error(f"Pinecone index '{index_name}' not found. Available indexes: {index_list}")
                    return None
                return pc.Index(index_name)
            except Exception as e:
                st.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
                raise

        video_index = connect_to_index(video_index_name)
        audio_index = connect_to_index(audio_index_name)
        text_index = connect_to_index(text_index_name)
        desc_index = connect_to_index(desc_index_name)
        
        return video_index, audio_index, text_index, desc_index
    except Exception as e:
        error_msg = str(e)
        if "network" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            st.error(f"Pinecone connection failed due to network error: {e}. Please check your internet connection and try again.")
        else:
            st.error(f"Pinecone connection failed: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None, None, None, None

# Initialize clients if config is loaded
if config:
    s3_client = init_minio_client(config)
    mongo_collection = init_mongo_collection(config)
    video_pinecone_index, audio_pinecone_index, text_pinecone_index, desc_pinecone_index = init_pinecone_indexes(config)
    # Initialize OpenAI client if configured
    if config.get("openai") and config["openai"].get("api_key"):
        try:
            openai_client = openai.OpenAI(api_key=config["openai"]["api_key"])
        except Exception as e:
            st.warning(f"Could not initialize OpenAI client: {e}")
            openai_client = None
    else:
        # Debug: Check if secrets are available
        try:
            if "openai" in st.secrets:
                st.warning("OpenAI secret found but not loaded into config. Check secrets.toml format.")
            else:
                st.warning("OpenAI secret not found in st.secrets. Make sure secrets.toml is in .streamlit/ directory.")
        except:
            pass
        openai_client = None
else:
    s3_client, mongo_collection, video_pinecone_index, audio_pinecone_index, text_pinecone_index, desc_pinecone_index, openai_client = None, None, None, None, None, None, None

# --- Main App Logic Starts Here ---

if s3_client is None or mongo_collection is None or video_pinecone_index is None or audio_pinecone_index is None or text_pinecone_index is None or desc_pinecone_index is None:
    st.warning("Application is not fully configured. Please check your secrets and service availability.")
    st.stop()

# --- Model Loading ---

@st.cache_resource
def load_imagebind_model():
    """Loads the ImageBind model and caches it."""
    # Try GPU first, but fallback to CPU if OOM
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Define the path to the local checkpoint (relative to project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(project_root, ".checkpoints", "imagebind_huge.pth")
    
    status_placeholder = st.empty()

    try:
        # Clear CUDA cache before loading to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Check available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            free_memory = total_memory - reserved_memory
            
            if free_memory < 2 * 1024**3:  # Less than 2GB free
                st.warning(
                    f"GPU memory is low ({free_memory / 1024**3:.2f} GB free, "
                    f"{reserved_memory / 1024**3:.2f} GB reserved). "
                    f"Consider freeing GPU memory or the model will fallback to CPU. "
                    f"You can also set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation."
                )
        
        if not os.path.exists(checkpoint_path):
            status_placeholder.info(f"Model checkpoint not found at '{checkpoint_path}'. Downloading model weights...")
            st.warning("Falling back to downloading the model...")
            # Load to CPU first to avoid OOM during download
            model = imagebind_model.imagebind_huge(pretrained=True)
            status_placeholder.empty()
        else:
            status_placeholder.info("Loading ImageBind model from local checkpoint...")
            # Create model architecture first
            model = imagebind_model.imagebind_huge(pretrained=False)
            # Load checkpoint to CPU first to avoid OOM, then move to device
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            # Clear the loading message once done
            status_placeholder.empty()

        model.eval()
        
        # Try to move to device, fallback to CPU if OOM
        try:
            model.to(device)
            if device.startswith('cuda'):
                # Verify model is actually on GPU
                next(model.parameters()).device
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if device.startswith('cuda'):
                st.warning(f"CUDA out of memory. Falling back to CPU. Error: {e}")
                device = "cpu"
                torch.cuda.empty_cache()  # Clear GPU cache
                model.to(device)
            else:
                raise
        
        return model, device
        
    except Exception as e:
        status_placeholder.error(f"Failed to load ImageBind model: {e}")
        # Try CPU as last resort
        if device.startswith('cuda'):
            try:
                st.warning("Attempting to load model on CPU as fallback...")
                device = "cpu"
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model = imagebind_model.imagebind_huge(pretrained=False)
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'], strict=False)
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    model = imagebind_model.imagebind_huge(pretrained=True)
                model.eval()
                model.to(device)
                st.info("Model loaded successfully on CPU.")
                return model, device
            except Exception as cpu_error:
                st.error(f"Failed to load model even on CPU: {cpu_error}")
                raise
        else:
            raise

model, device = load_imagebind_model()

@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    """Loads a Whisper model and caches it."""
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        st.error(f"Could not load Whisper model: {e}")
        return None

whisper_model = load_whisper_model()


@st.cache_resource
def load_captioning_model():
    """Loads the BLIP image captioning model and processor."""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return processor, model
    except Exception as e:
        st.error(f"Could not load captioning model: {e}")
        return None, None

caption_processor, caption_model = load_captioning_model()
if caption_model and caption_processor and torch.cuda.is_available():
    caption_model.to("cuda:0")



# --- Video Processing and Embedding Logic ---

def transcribe_video_with_whisper(
    video_path: str, 
    whisper_model
) -> Tuple[list, str | None]:
    """
    Extracts audio from a video, saves it as a WAV file, and transcribes it.
    Returns the transcript segments and the path to the persistent WAV file.
    """
    if not whisper_model or not os.path.exists(video_path):
        return [], None

    # Create a persistent temporary file that we can clean up later
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = temp_audio_file.name
    temp_audio_file.close()  # Close the file handle so ffmpeg can write to it
    
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", "-f", "wav", "-y", audio_path, "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 44:
            # Use Whisper for transcription
            # task="translate" will transcribe and translate non-English audio.
            result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available(), task='translate')
            return result.get("segments", []), audio_path
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"Error during full audio transcription: {e}", file=sys.stderr)
        # Clean up the temp file on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return [], None
    
    # If something went wrong before transcription but after file creation
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return [], None

def transcribe_audio(
    audio_path: str, 
    whisper_model
) -> str:
    """Transcribes a standalone audio file and returns the full text."""
    if not whisper_model or not os.path.exists(audio_path):
        return ""
    
    # Use ffmpeg to convert to a standardized format for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
        wav_path = temp_wav.name
        cmd = [
            "ffmpeg", "-i", audio_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", "-y", wav_path, "-loglevel", "error"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                # Use Whisper for transcription
                result = whisper_model.transcribe(wav_path, fp16=torch.cuda.is_available(), task='translate')
                return result.get("text", "").strip()
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            print(f"Error during standalone audio transcription: {e}", file=sys.stderr)
            return ""
    return ""

def find_segment_by_scene_end(scene_end_time: float, segments: list, tolerance: float = 0.5) -> dict:
    """
    Finds the transcript segment whose 'end' time matches the scene's end time.
    Uses adaptive tolerance based on scene characteristics for better matching.
    
    Args:
        scene_end_time: End time of the detected scene in seconds
        segments: List of transcript segments from Whisper
        tolerance: Base tolerance in seconds (default: 0.5, reduced for better short scene matching)
    
    Returns:
        Segment dictionary if found, None otherwise
    """
    if not segments:
        return None
    
    # First, try to find exact match or very close match within tolerance
    best_match = None
    min_diff = float('inf')
    
    for segment in segments:
        segment_end = segment.get('end', 0)
        diff = abs(scene_end_time - segment_end)
        
        # Use adaptive tolerance: for very short scenes, use tighter tolerance
        # For longer scenes, use the base tolerance
        adaptive_tolerance = min(tolerance, max(0.1, scene_end_time * 0.1))
        
        if diff < adaptive_tolerance and diff < min_diff:
            min_diff = diff
            best_match = segment
    
    # If no match within tolerance, find the segment that ends just before or at the scene end
    # This handles cases where scene ends between transcript segments
    if best_match is None:
        for segment in segments:
            segment_end = segment.get('end', 0)
            if segment_end <= scene_end_time:
                diff = scene_end_time - segment_end
                if diff < min_diff:
                    min_diff = diff
                    best_match = segment
    
    # If still no match, find the closest segment (before or after scene end)
    # This ensures we always get a transcript for the scene, even for very short scenes
    if best_match is None:
        for segment in segments:
            segment_end = segment.get('end', 0)
            diff = abs(scene_end_time - segment_end)
            if diff < min_diff:
                min_diff = diff
                best_match = segment
    
    return best_match

def extract_video_frame(video_path: str, timestamp_seconds: float) -> Image.Image:
    """
    Extracts a single frame from a video at a specific timestamp.
    Returns a PIL Image or None if extraction fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Convert timestamp to frame number
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Resize and convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return pil_image
        return None
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None

def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using multiple methods.
    Tries ffprobe first, then falls back to OpenCV.
    """
    if not os.path.exists(video_path):
        return 0.0
    
    # Method 1: Try ffprobe
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
        pass  # Fallback to OpenCV
    
    # Method 2: Fallback to OpenCV
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

def convert_to_mp4(input_path: str) -> str | None:
    """Converts a video file to MP4 format if it's not already."""
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == ".mp4":
        return input_path

    output_path = os.path.splitext(input_path)[0] + ".mp4"
    st.info(f"Converting video to MP4 for web playback...")
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vcodec", "libx264", "-acodec", "aac",
        "-pix_fmt", "yuv420p", # for compatibility
        "-y", output_path, "-loglevel", "error"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.info("✓ Conversion successful.")
            return output_path
        st.error(f"Conversion failed: output file is empty. FFMPEG stderr: {result.stderr}")
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to convert video to MP4. FFMPEG stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        st.error("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return None

def split_video_into_chunks(video_path: str, chunk_duration_seconds: float = 600.0) -> List[Tuple[float, float, str]]:
    """
    Split video into chunks of specified duration.
    """
    duration = get_video_duration(video_path)
    if duration == 0.0 or duration <= chunk_duration_seconds:
        return [(0.0, duration, video_path)]
    
    chunks = []
    chunk_dir = tempfile.mkdtemp(prefix="video_chunks_")
    start_time = 0.0
    
    for i in range(int(duration // chunk_duration_seconds) + 1):
        end_time = min(start_time + chunk_duration_seconds, duration)
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.mp4")
        
        # Use stream copy for speed, fallback to re-encode on failure
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-i", video_path,
            "-t", str(end_time - start_time), "-c", "copy", 
            "-avoid_negative_ts", "make_zero", "-y", chunk_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                cmd[9] = "libx264" # change codec to re-encode
                subprocess.run(cmd, capture_output=True, text=True, check=False)

            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                chunks.append((start_time, end_time, chunk_path))
        except Exception:
            pass # Skip failed chunk
        start_time = end_time

    return chunks if chunks else [(0.0, duration, video_path)]

def extract_frames_parallel(video_path: str, scene_list: list, fps: float, chunk_start_offset: float = 0.0) -> Dict[int, Tuple[np.ndarray, Dict[str, float]]]:
    """Extracts middle frames from scenes in parallel."""
    def extract_single_frame(args):
        scene_idx, frame_num, start_frame, end_frame = args
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return scene_idx, None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame_np = cap.read()
            cap.release()
            if ret and frame_np is not None:
                timestamps = {
                    "start_time": chunk_start_offset + (start_frame / fps),
                    "end_time": chunk_start_offset + (end_frame / fps),
                }
                return scene_idx, (frame_np, timestamps)
            return scene_idx, None
        except Exception:
            return scene_idx, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = [
            (i, s[0].get_frames() + (s[1].get_frames() - s[0].get_frames()) // 2, s[0].get_frames(), s[1].get_frames())
            for i, s in enumerate(scene_list)
        ]
        results = dict(executor.map(extract_single_frame, args))
        return {k: v for k, v in results.items() if v is not None}

def get_video_properties(video_path: str) -> dict:
    """Extracts properties like duration, fps, resolution from a video file."""
    properties = {
        "duration": 0.0,
        "fps": 0,
        "frame_count": 0,
        "width": 0,
        "height": 0
    }
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            properties["fps"] = cap.get(cv2.CAP_PROP_FPS)
            properties["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            properties["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            properties["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if properties["fps"] > 0:
                properties["duration"] = properties["frame_count"] / properties["fps"]
            cap.release()
    except Exception as e:
        st.error(f"Failed to get video properties: {e}")
    return properties

def detect_scenes_with_adaptive_detector(video_path: str) -> list:
    """
    Detect scenes using AdaptiveDetector with the exact same configuration as app_adaptive.py.
    Returns all detected scenes without any minimum duration filtering.
    
    Args:
        video_path: Path to video file
    
    Returns:
        List of all detected scenes from SceneManager
    """
    video_stream = None
    # Use context manager for stderr redirection to avoid thread-safety issues
    # This ensures stderr is properly restored even if exceptions occur
    with open(os.devnull, 'w') as devnull:
        with redirect_stderr(devnull):
            try:
                video_stream = open_video(video_path)
                scene_manager = SceneManager()
                # Use AdaptiveDetector with adaptive threshold 3 and custom weights
                # weights: (hue, saturation, luminosity, edges) = (1, 1, 5, 0)
                # Create Components object with custom weights
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
                st.warning(f"Scene detection failed for {os.path.basename(video_path)}: {e}")
                import traceback
                traceback.print_exc()
                return []
            finally:
                # Try to close video stream if it has a close method
                if video_stream:
                    try:
                        if hasattr(video_stream, 'close'):
                            video_stream.close()
                    except Exception:
                        pass

def resize_frame_optimized(frame_np, max_size: int = 512) -> Image.Image:
    """
    Resize frame to reduce processing time while maintaining aspect ratio.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    return pil_image

def get_batch_embeddings(
    pil_images: List[Image.Image],
    audio_paths: List[str],
    texts: List[str],
    device: str,
    model
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates embeddings for a batch of images, audio clips, and texts."""
    image_embeddings = torch.empty((0,))
    audio_embeddings = torch.empty((0,))
    text_embeddings = torch.empty((0,))

    # Process images if any
    if pil_images:
        temp_image_paths = []
        try:
            for pil_image in pil_images:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    pil_image.save(temp_file, format='PNG')
                    temp_image_paths.append(temp_file.name)
            
            vision_inputs = {ModalityType.VISION: data.load_and_transform_vision_data(temp_image_paths, device)}
            with torch.no_grad():
                image_embeddings = model(vision_inputs)[ModalityType.VISION]
        finally:
            for path in temp_image_paths:
                try: os.remove(path)
                except OSError: pass

    # Process audio if any
    if audio_paths:
        try:
            # Filter out empty or invalid audio paths
            valid_audio_paths = [path for path in audio_paths if path and os.path.exists(path)]
            if valid_audio_paths:
                audio_inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(valid_audio_paths, device)}
                with torch.no_grad():
                    audio_embeddings = model(audio_inputs)[ModalityType.AUDIO]
        except (ValueError, OSError, Exception) as e:
            # Handle various errors including logging errors from ImageBind
            error_msg = str(e)
            if "I/O operation on closed file" in error_msg or "logging" in error_msg.lower():
                # Silently skip if it's a logging/stderr issue - audio embeddings will be empty
                pass
            else:
                print(f"Error getting audio embeddings: {e}", file=sys.stderr)
            
    # Process text if any
    if texts:
        try:
            text_inputs = {ModalityType.TEXT: data.load_and_transform_text(texts, device)}
            with torch.no_grad():
                text_embeddings = model(text_inputs)[ModalityType.TEXT]
        except Exception as e:
            print(f"Error getting text embeddings: {e}")

    return image_embeddings, audio_embeddings, text_embeddings


def generate_scene_descriptions(pil_images: List[Image.Image], processor, model, device: str) -> List[str]:
    """Generates text descriptions for a batch of image frames using BLIP."""
    if not processor or not model or not pil_images:
        return [""] * len(pil_images)
    try:
        # Use a consistent device for model and inputs
        model_device = next(model.parameters()).device
        inputs = processor(images=pil_images, return_tensors="pt").to(model_device)
        
        generated_ids = model.generate(**inputs, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [caption.strip() for caption in generated_captions]
    except Exception as e:
        print(f"Error during batch caption generation: {e}", file=sys.stderr)
        return [""] * len(pil_images)


class PineconeUploadManager:
    """Manages background uploads to Pinecone using a queue and worker threads."""
    def __init__(self, index, num_workers: int = 4, batch_size: int = 100):
        self.index = index
        self.batch_size = batch_size
        self.upload_queue = queue.Queue()
        self.workers = []
        self.is_running = True
        self.uploaded_count = 0
        self.errors = []
        self.lock = threading.Lock()
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
                        # Retry logic for network errors
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
                                # Check if it's a network-related error
                                if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'axios', 'http', 'request']) and retry_count < max_retries - 1:
                                    retry_count += 1
                                    sleep(2.0 * retry_count)  # Exponential backoff
                                    continue
                                # If it's not a network error or we've exhausted retries, log the error
                                with self.lock:
                                    self.errors.append(f"Worker {worker_id}: {e}")
                                break
                finally:
                    self.upload_queue.task_done()
            except queue.Empty:
                continue

    def add_to_queue(self, vectors: list):
        if vectors:
            self.upload_queue.put(vectors)

    def wait_for_completion(self):
        self.upload_queue.join()

    def stop(self):
        self.is_running = False
        for worker in self.workers:
            worker.join(timeout=2)
    
    def get_stats(self):
        with self.lock:
            return {"uploaded": self.uploaded_count, "errors": len(self.errors)}

def prepare_scene_data(args):
    """
    Prepares all necessary data for a single scene for batch processing.
    This function is designed to be run in a thread pool.
    """
    chunk_path, scene, fps, chunk_start_offset, video_name, video_doc_id, global_scene_index, transcript_segments, full_audio_segment = args
    
    abs_start_time = chunk_start_offset + scene[0].get_seconds()
    abs_end_time = chunk_start_offset + scene[1].get_seconds()

    # --- Frame Extraction ---
    start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
    middle_frame = start_frame + (end_frame - start_frame) // 2
    
    cap = cv2.VideoCapture(chunk_path)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame_np = cap.read()
    cap.release()
    if not ret: return None
    
    # --- Audio Slicing (Thread-Safe) ---
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
            scene_audio_path = "" # Fail silently for a single clip
    
    return {
        "pil_image": resize_frame_optimized(frame_np),
        "audio_path": scene_audio_path,
        "transcript": find_segment_by_scene_end(abs_end_time, transcript_segments).get('text', '').strip(),
        "video_name": video_name,
        "video_doc_id": video_doc_id,
        "scene_global_index": global_scene_index,
        "abs_start_time": abs_start_time,
        "abs_end_time": abs_end_time,
    }

def process_batch(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes a batch of scene data to generate descriptions and embeddings.
    """
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
        model=model
    )
    
    _, _, desc_embeddings = get_batch_embeddings(
        pil_images=[], 
        audio_paths=[], 
        texts=descriptions,
        device=device, 
        model=model
    )

    for i, item in enumerate(batch_data):
        scene_uuid = str(uuid.uuid4())
        description = descriptions[i]
        metadata = {
            "video_name": item['video_name'], 
            "video_doc_id": str(item['video_doc_id']),
            "scene_index": item['scene_global_index'], 
            "start_time": item['abs_start_time'],
            "end_time": item['abs_end_time'], 
            "transcript": item['transcript'],
            "description": description, 
            "scene_uuid": scene_uuid
        }

        video_vector = None
        if image_embeddings.nelement() > 0 and i < len(image_embeddings):
            video_vector = {"id": scene_uuid, "values": image_embeddings[i].cpu().numpy().tolist(), "metadata": metadata}

        audio_vector = None
        if audio_embeddings.nelement() > 0 and i < len(audio_embeddings):
            audio_vector = {"id": scene_uuid, "values": audio_embeddings[i].cpu().numpy().tolist(), "metadata": metadata}
            
        text_vector = None
        if text_embeddings.nelement() > 0 and i < len(text_embeddings):
            text_vector = {"id": scene_uuid, "values": text_embeddings[i].cpu().numpy().tolist(), "metadata": metadata}

        desc_vector = None
        if desc_embeddings.nelement() > 0 and i < len(desc_embeddings):
            desc_vector = {"id": scene_uuid, "values": desc_embeddings[i].cpu().numpy().tolist(), "metadata": metadata}

        mongo_data = {
            "scene_index": item['scene_global_index'], 
            "start_time": item['abs_start_time'],
            "end_time": item['abs_end_time'], 
            "transcript": item['transcript'], 
            "description": description, 
            "analysis": {},  # OpenAI analysis was removed
            "scene_uuid": scene_uuid
        }
        
        results.append({
            "video_vector": video_vector,
            "audio_vector": audio_vector,
            "text_vector": text_vector,
            "desc_vector": desc_vector,
            "mongo_data": mongo_data
        })
        
    return results


def process_video_chunk(
    args: Tuple
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Processes a single video chunk to detect scenes, prepare data, and generate embeddings.
    Designed to be run in a parallel ThreadPoolExecutor.
    Returns a list of scene data for MongoDB and a list of temporary audio clip paths for cleanup.
    """
    (
        chunk_path,
        chunk_start_offset,
        video_name,
        video_doc_id,
        chunk_scene_offset,
        transcript_segments,
        full_audio_segment,
        video_upload_manager,
        audio_upload_manager,
        text_upload_manager,
        desc_upload_manager,
    ) = args

    # --- 1. Detect scenes for this chunk ---
    scene_list = detect_scenes_with_adaptive_detector(chunk_path)
    if not scene_list:
        return [], []

    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # --- 2. Prepare data for all scenes in this chunk in parallel ---
    scene_args = [
        (chunk_path, scene, fps, chunk_start_offset, video_name, video_doc_id, chunk_scene_offset + i, transcript_segments, full_audio_segment)
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

    # --- 3. Process the prepared scene data in batches ---
    all_scene_data_mongo = []
    for i in range(0, len(chunk_scene_data), BATCH_SIZE):
        batch = chunk_scene_data[i:i + BATCH_SIZE]
        batch_results = process_batch(batch)

        for result in batch_results:
            if result['video_vector']:
                video_upload_manager.add_to_queue([result['video_vector']])
            if result['audio_vector']:
                audio_upload_manager.add_to_queue([result['audio_vector']])
            if result['text_vector']:
                text_upload_manager.add_to_queue([result['text_vector']])
            if result['desc_vector']:
                desc_upload_manager.add_to_queue([result['desc_vector']])
            if result['mongo_data']:
                all_scene_data_mongo.append(result['mongo_data'])

    return all_scene_data_mongo, temp_audio_clips


def process_video_and_generate_embeddings(video_path: str, video_doc_id: str, video_name: str, minio_key: str, status_cb, whisper_model):
    """The main processing pipeline for a single video."""
    audio_path = None
    temp_audio_clips = []
    all_scene_data_mongo = []
    
    try:
        status_cb("Step 1/6: Extracting video properties...")
        properties = get_video_properties(video_path)
        properties["duration"] = get_video_duration(video_path)
        mongo_collection.update_one({"_id": video_doc_id}, {"$set": {
            "video_duration": properties["duration"], "frame_rate": properties["fps"],
            "resolution": f"{properties['width']}x{properties['height']}",
            "processing_status": "processing"
        }})

        status_cb("Step 2/6: Transcribing full video audio...")
        transcript_segments, audio_path = transcribe_video_with_whisper(video_path, whisper_model)
        if not transcript_segments:
            status_cb("Warning: Could not generate transcript. Audio processing will be skipped.")

        # Load full audio for slicing if it exists
        full_audio_segment = None
        if audio_path and os.path.exists(audio_path):
            try:
                full_audio_segment = AudioSegment.from_wav(audio_path)
            except Exception as e:
                status_cb(f"Warning: Could not load audio for slicing: {e}")

        status_cb("Step 3/6: Splitting video into chunks...")
        chunks = split_video_into_chunks(video_path)
        is_chunked = len(chunks) > 1 and chunks[0][2] != video_path
        
        status_cb(f"Step 3/6: Processing {len(chunks)} video chunk(s) in parallel...")
        
        # Create shared upload managers for all chunks
        video_upload_manager = PineconeUploadManager(video_pinecone_index)
        audio_upload_manager = PineconeUploadManager(audio_pinecone_index)
        text_upload_manager = PineconeUploadManager(text_pinecone_index)
        desc_upload_manager = PineconeUploadManager(desc_pinecone_index)

        total_scenes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            # Prepare arguments for each chunk
            chunk_args_list = []
            chunk_scene_offset = 0
            
            # Pre-calculate scene counts to determine global scene index offsets
            scene_counts = [len(detect_scenes_with_adaptive_detector(c[2])) for c in chunks]

            for i, (chunk_start, _, chunk_path) in enumerate(chunks):
                num_scenes_in_chunk = scene_counts[i]
                args = (
                    chunk_path, chunk_start, video_name, video_doc_id, chunk_scene_offset,
                    transcript_segments, full_audio_segment,
                    video_upload_manager, audio_upload_manager, text_upload_manager, desc_upload_manager
                )
                chunk_args_list.append(args)
                chunk_scene_offset += num_scenes_in_chunk
            
            total_scenes = chunk_scene_offset
            status_cb(f"Step 4/6: Detected a total of {total_scenes} scenes. Starting parallel processing...")

            # Submit chunk processing tasks
            future_to_chunk = {executor.submit(process_video_chunk, arg): arg for arg in chunk_args_list}
            
            processed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                mongo_data, audio_files = future.result()
                if mongo_data:
                    all_scene_data_mongo.extend(mongo_data)
                if audio_files:
                    temp_audio_clips.extend(audio_files)
                processed_chunks += 1
                status_cb(f"Step 5/6: Completed processing chunk {processed_chunks}/{len(chunks)}...")

        if not all_scene_data_mongo:
             status_cb("No scenes were processed successfully.")
             mongo_collection.update_one({"_id": video_doc_id}, {"$set": {"processing_status": "completed", "scenes_detected": 0}})
             return

        status_cb("Step 6/6: Finalizing data and cleaning up...")
        if transcript_segments:
            transcript_minio_key = f"{os.path.splitext(minio_key)[0]}.json"
            transcript_data = json.dumps(transcript_segments, indent=4).encode('utf-8')
            
            @retry_on_network_error(max_retries=3, delay=1.0)
            def upload_transcript():
                s3_client.put_object(
                    Bucket=config["minio"]["transcripts_bucket_name"],
                    Key=transcript_minio_key,
                    Body=BytesIO(transcript_data),
                    ContentLength=len(transcript_data),
                    ContentType='application/json'
                )
            
            try:
                upload_transcript()
                # Store reference in mongo
                mongo_collection.update_one(
                    {"_id": video_doc_id},
                    {"$set": {"transcript_minio_key": transcript_minio_key}}
                )
            except Exception as e:
                status_cb(f"Warning: Failed to upload transcript to MinIO: {e}")

        video_upload_manager.wait_for_completion()
        audio_upload_manager.wait_for_completion()
        text_upload_manager.wait_for_completion()
        desc_upload_manager.wait_for_completion()
        video_upload_manager.stop()
        audio_upload_manager.stop()
        text_upload_manager.stop()
        desc_upload_manager.stop()
        
        if is_chunked:
            for _, _, chunk_path in chunks:
                if video_path != chunk_path:
                    try: os.remove(chunk_path)
                    except OSError: pass
            try: os.rmdir(os.path.dirname(chunks[0][2]))
            except OSError: pass

        video_stats = video_upload_manager.get_stats()
        audio_stats = audio_upload_manager.get_stats()
        text_stats = text_upload_manager.get_stats()
        desc_stats = desc_upload_manager.get_stats()
        mongo_collection.update_one({"_id": video_doc_id}, {"$set": {
            "processing_status": "completed", 
            "video_embeddings_count": video_stats["uploaded"],
            "audio_embeddings_count": audio_stats["uploaded"],
            "text_embeddings_count": text_stats["uploaded"],
            "desc_embeddings_count": desc_stats["uploaded"]
        }})
        if all_scene_data_mongo:
            mongo_collection.update_one(
                {"_id": video_doc_id},
                {"$push": {"scenes": {"$each": sorted(all_scene_data_mongo, key=lambda x: x['scene_index'])}}}
            )
        status_cb(f"Processing complete! {video_stats['uploaded']} video, {audio_stats['uploaded']} audio, {text_stats['uploaded']} text, and {desc_stats['uploaded']} description embeddings uploaded.")

    except Exception as e:
        mongo_collection.update_one({"_id": video_doc_id}, {"$set": {"processing_status": "failed", "error_message": str(e)}})
        status_cb(f"Processing failed: {e}")
        raise
    finally:
        # --- Final Cleanup ---
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


def merge_overlapping_clips(search_results: list, gap_seconds: int = 3) -> list:
    """Merges overlapping or nearby video clips from the same video."""
    if not search_results:
        return []

    # Group results by video
    results_by_video = {}
    for res in search_results:
        video_id = res.metadata.get("video_doc_id")
        if video_id not in results_by_video:
            results_by_video[video_id] = []
        results_by_video[video_id].append(res)

    final_merged_results = []
    for video_id, clips in results_by_video.items():
        if not clips:
            continue

        # Sort clips by start time to easily check for overlaps
        clips.sort(key=lambda x: x.metadata['start_time'])

        merged_for_video = []
        if not clips:
            continue
        current_merge = clips[0]

        for next_clip in clips[1:]:
            # Check for overlap or if the next clip is within the gap
            if next_clip.metadata['start_time'] <= current_merge.metadata['end_time'] + gap_seconds:
                # Merge clips
                current_merge.metadata['end_time'] = max(current_merge.metadata['end_time'], next_clip.metadata['end_time'])
                
                # Keep the highest score
                if next_clip.score > current_merge.score:
                    current_merge.score = next_clip.score
                
                # Combine source information
                current_source = set(current_merge.metadata.get('source', '').split(', '))
                next_source = set(next_clip.metadata.get('source', '').split(', '))
                # Filter out empty strings that might result from splitting an empty string
                current_source.discard('')
                next_source.discard('')
                
                combined_sources = sorted(list(current_source.union(next_source)))
                current_merge.metadata['source'] = ', '.join(combined_sources)
            else:
                # No overlap, finalize the current merged clip and start a new one
                merged_for_video.append(current_merge)
                current_merge = next_clip
        
        merged_for_video.append(current_merge) # Add the last merged clip
        final_merged_results.extend(merged_for_video)

    # Sort the final results by score again
    final_merged_results.sort(key=lambda x: x.score, reverse=True)
    return final_merged_results


def render_search_results_grid(search_results, num_columns=4, original_count=None):
    """Renders search results in a grid layout."""
    
    if not search_results:
        st.warning("No matching scenes found.")
        return

    merged_count = len(search_results)
    if original_count is not None and original_count > merged_count:
        st.success(f"Found {merged_count} matching scenes")
        # (merged from {original_count} results).
    else:
        st.success(f"Found {merged_count} matching scenes.")
    
    # Custom CSS for score badges
    st.markdown("""
        <style>
        .score-badge {
            display: inline-block;
            padding: .25em .4em;
            font-size: 75%;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: .25rem;
        }
        .score-high { color: #fff; background-color: #28a745; }
        .score-medium { color: #212529; background-color: #ffc107; }
        .score-low { color: #fff; background-color: #6c757d; }
        .source-badge {
            color: #fff;
            background-color: #17a2b8;
            margin-left: 4px;
        }
        .video-container { margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(num_columns)
    for i, match in enumerate(search_results):
        col = cols[i % num_columns]
        with col:
            with st.container():
                meta = match.metadata
                score = match.score

                if score >= 0.5:
                    level, badge_class = "HIGH", "score-high"
                elif score >= 0.3:
                    level, badge_class = "MEDIUM", "score-medium"
                else:
                    level, badge_class = "LOW", "score-low"

                try:
                    video_doc_id = ObjectId(meta["video_doc_id"])
                    video_doc = mongo_collection.find_one({"_id": video_doc_id})
                    
                    if video_doc:
                        video_bytes = get_video_bytes(video_doc)
                        if not video_bytes:
                            continue # Skip if video can't be fetched
                        
                        st.markdown(f"**{meta['video_name']}**")
                        
                        start_ts = time.strftime('%H:%M:%S', time.gmtime(meta['start_time']))
                        end_ts = time.strftime('%H:%M:%S', time.gmtime(meta['end_time']))
                        source_text = meta.get('source', 'N/A')
                        
                        html_string = f"<div class='video-container'>"
                        
                        # Display quality match badge. The score part is commented out.
                        score_text = "" # f" ({score:.2f})" 
                        html_string += f"<span class='score-badge {badge_class}'>{level}{score_text}</span>"
                        
                        # The source badge is also commented out.
                        # source_badge_text = f"<span class='score-badge source-badge'>{source_text}</span>"
                        # html_string += source_badge_text

                        html_string += f" <span>{start_ts} - {end_ts}</span>"
                        html_string += f"</div>"
                        st.markdown(html_string, unsafe_allow_html=True)

                        # Display video starting at the detected scene time
                        try:
                            st.video(video_bytes, start_time=int(meta['start_time']))
                        except Exception as video_error:
                            st.error(f"Video playback error: {video_error}")
                            # Show info as fallback for debugging
                            with st.expander("Debug: Video Info"):
                                st.markdown(f"**Start time:** {start_ts} | **End time:** {end_ts}")
                        
                except (ClientError, ValueError, KeyError) as e:
                    st.error(f"Could not display result: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                except Exception as e:
                    st.error(f"Unexpected error displaying result: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())


def compute_final_score(modality_scores, threshold=0.75):
    """
    modality_scores: dict like {'video': 0.33, 'audio': 0.36, ...}
    returns: final weighted score (float)
    """
    # ---- 1️⃣  Check for any raw score ≥ threshold ----
    if not modality_scores:
        return 0.0
    max_raw_score = max(modality_scores.values())
    if max_raw_score >= threshold:
        return round(max_raw_score, 3)   # use raw score directly

    # ---- 2️⃣  Compute base fused score (weighted average) ----
    weights = {"video": 1.0, "desc": 0.9, "audio": 0.8, "text": 0.6}
    
    # Filter out modalities not present in the scores
    present_modalities = [m for m in weights if m in modality_scores]
    if not present_modalities:
        return 0.0

    total_w = sum(weights[m] for m in present_modalities)
    base_score = sum(modality_scores[m] * weights[m] for m in present_modalities) / total_w

    # ---- 3️⃣  Identify top-priority modality present ----
    priority_order = ["video", "desc", "audio", "text"]
    top_mod = next((m for m in priority_order if m in present_modalities), None)
    top_weight = weights[top_mod] if top_mod else 0.6  # fallback if something odd

    # ---- 4️⃣  Apply priority boost rule (for low base scores) ----
    # This ensures low scores get lifted based on top modality importance
    final_score = base_score * 0.7 + top_weight * 0.3
    return round(final_score, 3)


def compute_audio_search_score(modality_scores):
    """
    Computes a final score specifically for audio-based searches.
    Priority is given to audio similarity, then to text similarity.
    """
    audio_score = modality_scores.get('audio', 0.0)
    text_score = modality_scores.get('text', 0.0)

    if audio_score >= 0.75:
        return audio_score
    
    final_score = (audio_score * 1.0 + text_score * 0.4) / 1.4
    return round(final_score, 3)


def perform_audio_search(temp_file_path):
    """
    Performs a specialized search for an uploaded audio file.
    Returns a list of final, scored search results.
    """
    search_tasks = {}
    
    # 1. Generate audio embedding and prepare audio search task
    audio_embedding_tensor, _, _ = get_batch_embeddings(pil_images=[], audio_paths=[temp_file_path], texts=[], device=device, model=model)
    if audio_embedding_tensor.nelement() > 0:
        search_tasks[audio_pinecone_index] = audio_embedding_tensor[0].cpu().numpy().tolist()

    # 2. Transcribe audio, generate text embedding, and prepare text search task
    transcript = transcribe_audio(
        temp_file_path, 
        whisper_model
    )
    if transcript:
        st.info(f"Extracted transcript: \"{transcript}\"")
        _, _, text_embedding_tensor = get_batch_embeddings(pil_images=[], audio_paths=[], texts=[transcript], device=device, model=model)
        if text_embedding_tensor.nelement() > 0:
            search_tasks[text_pinecone_index] = text_embedding_tensor[0].cpu().numpy().tolist()

    if not search_tasks:
        return []

    # 3. Query Pinecone indexes
    results_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(query_index, index, vec): index for index, vec in search_tasks.items()}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results_map[index] = future.result()

    # 4. Process and score results
    merged_results = {}
    
    index_to_name = {
        audio_pinecone_index: 'audio',
        text_pinecone_index: 'text'
    }

    for index, results in results_map.items():
        source_name = index_to_name.get(index)
        if source_name and results:
            for match in results.matches:
                scene_uuid = match.metadata.get("scene_uuid")
                if not scene_uuid: continue
                
                if scene_uuid not in merged_results:
                    merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
                else:
                    merged_results[scene_uuid]['scores'][source_name] = match.score
                    if match.score > merged_results[scene_uuid]['match'].score:
                        merged_results[scene_uuid]['match'] = match

    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        best_match.score = compute_audio_search_score(scores_dict)
        final_results.append(best_match)

    return sorted(final_results, key=lambda x: x.score, reverse=True)


def get_transcript_segments(video_doc: dict) -> list | None:
    """Downloads transcript from MinIO and returns it as a list of segments."""
    if not video_doc or 'transcript_minio_key' not in video_doc:
        return None
    try:
        @retry_on_network_error(max_retries=3, delay=1.0)
        def download_transcript():
            transcript_obj = s3_client.get_object(
                Bucket=config["minio"]["transcripts_bucket_name"], 
                Key=video_doc['transcript_minio_key']
            )
            return json.loads(transcript_obj['Body'].read())
        return download_transcript()
    except Exception as e:
        st.error(f"Failed to download transcript: {e}")
        return None

def analyze_video_with_openai(video_name: str, prompt: str):
    """
    Analyzes a video using OpenAI by retrieving transcripts and sending them with the user prompt.
    Returns the response text or None if there's an error.
    """
    if not openai_client:
        st.error("OpenAI client is not configured. Please add your API key to secrets.")
        return None

    try:
        # Get the video document to retrieve video_doc_id
        video_doc = mongo_collection.find_one({"original_filename": video_name})
        if not video_doc:
            st.error(f"Video '{video_name}' not found in database.")
            return None
        
        video_doc_id = video_doc["_id"]
        
        # Get transcript data from transcripts_collection
        segments = get_transcript_segments(video_doc)
        
        # Build transcript text from segments (PART2: new structure)
        # Format: end_time: text ; end_time: text (compact format to save tokens)
        transcript_text = ""
        if segments:
            transcript_parts = []
            for segment in segments:
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                if text:
                    # Format: end_time: text
                    transcript_parts.append(f"{end_time}: {text}")
            
            # Join with semicolon and space separator
            transcript_text = " ; ".join(transcript_parts)
        
        if not transcript_text:
            st.warning("No transcript found for this video. The video may not have been fully processed yet.")
            return None
        
        # Prepare the system prompt and user message
        system_prompt = """You are a video content analyst. You will receive a video transcript and a user's analysis request.
Provide a comprehensive, detailed response.
If you identify specific moments or clips in your analysis, list them with their start times in seconds.
Your response MUST be a JSON object with two keys:
1. "analysis_text": A string containing your full, detailed analysis.
2. "clips": A list of JSON objects, where each object represents a mentioned clip and has the keys "description" (string), and "start_time" (integer in seconds). If no specific clips are mentioned, provide an empty list.

Example of a clip object: {"description": "The speaker makes a key point about AI.", "start_time": 125}"""
        
        user_message = f"""Video Transcript:
{transcript_text}

User Request: {prompt}

Please provide a detailed analysis based on the above transcript and the user's request."""
        
        # Make OpenAI API call with retry logic
        @retry_on_network_error(max_retries=3, delay=2.0)
        def call_openai_api():
            return openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"},
                timeout=60.0  # Add timeout
            )
        
        with st.spinner("Analyzing video with OpenAI..."):
            try:
                response = call_openai_api()
                analysis_result = response.choices[0].message.content
                return json.loads(analysis_result)
            except Exception as e:
                error_msg = str(e)
                if "network" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                    st.error(f"Network error during OpenAI analysis: {e}. Please check your connection and try again.")
                else:
                    st.error(f"OpenAI analysis failed: {e}")
                return None
        
    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


def get_video_bytes(video_doc: dict) -> bytes | None:
    """Downloads video content from MinIO and returns it as bytes."""
    if not video_doc or 'minio_key' not in video_doc:
        return None
    try:
        @retry_on_network_error(max_retries=3, delay=1.0)
        def download_content():
            video_obj = s3_client.get_object(
                Bucket=config["minio"]["bucket_name"], 
                Key=video_doc['minio_key']
            )
            return video_obj['Body'].read()
        return download_content()
    except Exception as e:
        error_msg = str(e)
        if "network" in error_msg.lower() or "connection" in error_msg.lower():
            st.error(f"Network error downloading video: {e}. Please try again.")
        else:
            st.error(f"Failed to download video content for playback: {e}")
        return None


# --- Streamlit UI Layout ---

st.title("🎬 Video Prism: AI-Powered Video Search")

tab1, tab2, tab3 = st.tabs(["📹 Video Management", "🔍 Search", "📊 Analyze"])

# --- Video Management Tab ---
with tab1:
    st.header("Upload and Process Videos")

    # Initialize session state for tracking processed files
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'processing_in_progress' not in st.session_state:
        st.session_state.processing_in_progress = False

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv'],
        key="video_uploader"
    )

    if uploaded_file is not None:
        # Check if this file has already been processed in this session
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if video already exists in MongoDB
        existing_video = mongo_collection.find_one({"original_filename": uploaded_file.name})
        
        if existing_video:
            status = existing_video.get("processing_status", "unknown")
            if status == "completed":
                st.info(f"'{uploaded_file.name}' has already been processed. Status: {status}")
            elif status == "processing":
                st.warning(f"'{uploaded_file.name}' is currently being processed. Please wait...")
            elif status == "pending":
                st.info(f"'{uploaded_file.name}' is queued for processing. Click 'Process Video' to start.")
            else:
                st.info(f"'{uploaded_file.name}' exists with status: {status}")
            
            # Show option to reprocess
            if st.button("Reprocess Video", key=f"reprocess_{file_id}"):
                video_doc_id = existing_video["_id"]
                minio_key = existing_video["minio_key"]
                
                # Update status to processing
                mongo_collection.update_one(
                    {"_id": video_doc_id},
                    {"$set": {"processing_status": "processing"}}
                )
                
                status_placeholder = st.empty()
                def status_update(message):
                    status_placeholder.info(message)
                
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    @retry_on_network_error(max_retries=3, delay=2.0)
                    def download_file():
                        s3_client.download_fileobj(config["minio"]["bucket_name"], minio_key, temp_file)
                    try:
                        download_file()
                    except Exception as e:
                        st.error(f"Failed to download video from storage: {e}")
                        st.rerun()
                    
                    temp_file_path = temp_file.name
                
                status_update("Checking video format and converting if necessary...")
                converted_path = convert_to_mp4(temp_file_path)

                if not converted_path:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    st.stop()
                
                processing_path = converted_path

                # If conversion happened, re-upload to MinIO and update record
                if processing_path != temp_file_path:
                    status_update("Re-uploading converted file to storage...")
                    os.remove(temp_file_path)
                    new_minio_key = f"{os.path.splitext(minio_key)[0]}.mp4"
                    
                    # Use retry logic for upload
                    @retry_on_network_error(max_retries=3, delay=2.0)
                    def re_upload_converted_file():
                        s3_client.upload_file(processing_path, config["minio"]["bucket_name"], new_minio_key)

                    try:
                        re_upload_converted_file()
                        # Delete old file from MinIO if key has changed
                        if new_minio_key != minio_key:
                            s3_client.delete_object(Bucket=config["minio"]["bucket_name"], Key=minio_key)
                    except Exception as e:
                        st.error(f"Failed to re-upload converted video: {e}")
                        os.remove(processing_path)
                        st.rerun()

                    # Update mongo with new key and info
                    mongo_collection.update_one(
                        {"_id": video_doc_id},
                        {"$set": {
                            "minio_key": new_minio_key,
                            "content_type": "video/mp4",
                            "file_size": os.path.getsize(processing_path)
                        }}
                    )
                    minio_key = new_minio_key

                # --- Start timing the process ---
                start_time = time.time()
                
                process_video_and_generate_embeddings(processing_path, video_doc_id, uploaded_file.name, minio_key, status_update, whisper_model)
                
                # --- End timing and save duration ---
                end_time = time.time()
                duration = end_time - start_time
                mongo_collection.update_one(
                    {"_id": video_doc_id},
                    {"$set": {"processing_duration_seconds": duration}}
                )

                os.remove(processing_path)
                st.rerun()
        else:
            # New file - save locally, upload to MinIO, create metadata, and process automatically
            if file_id not in st.session_state.processed_files:
                
                status_placeholder = st.empty()
                def status_update(message):
                    status_placeholder.info(message)

                file_extension = os.path.splitext(uploaded_file.name)[1]
                temp_video_path = os.path.join(tempfile.gettempdir(), f"video_{uuid.uuid4()}{file_extension}")
                
                try:
                    status_update(f"Saving uploaded file locally...")
                    # Write file in chunks to avoid memory issues
                    with open(temp_video_path, 'wb') as f:
                        chunk_size = 1024 * 1024 * 10  # 10MB chunks
                        # Reset buffer to the beginning before reading
                        uploaded_file.seek(0)
                        while True:
                            chunk = uploaded_file.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)

                    # Convert to MP4 for web playback
                    converted_path = convert_to_mp4(temp_video_path)
                    if not converted_path:
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        st.stop()

                    if converted_path != temp_video_path:
                        os.remove(temp_video_path)
                    processing_path = converted_path

                    status_update(f"Uploading and processing '{uploaded_file.name}'...")
                    
                    # Create a unique key for MinIO, ensuring .mp4 extension
                    minio_key = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}.mp4"
                    
                    # Upload to MinIO from the temp file path with retry logic
                    @retry_on_network_error(max_retries=3, delay=2.0)
                    def upload_to_minio():
                        s3_client.upload_file(processing_path, config["minio"]["bucket_name"], minio_key)
                    
                    try:
                        upload_to_minio()
                    except Exception as e:
                        error_msg = str(e)
                        if "network" in error_msg.lower() or "connection" in error_msg.lower():
                            status_placeholder.error(f"Network error uploading to storage: {e}. Please check your connection and try again.")
                        else:
                            status_placeholder.error(f"Failed to upload to storage: {e}")
                        if 'video_doc_id' in locals() and video_doc_id:
                            mongo_collection.update_one(
                                {"_id": video_doc_id},
                                {"$set": {"processing_status": "failed", "error_message": str(e)}}
                            )
                        if os.path.exists(processing_path):
                            os.remove(processing_path)
                    
                    # Create metadata record in MongoDB
                    metadata = {
                        "video_id": str(uuid.uuid4()),
                        "original_filename": uploaded_file.name,
                        "minio_key": minio_key,
                        "content_type": "video/mp4",
                        "file_size": os.path.getsize(processing_path),
                        "upload_timestamp": datetime.utcnow(),
                        "processing_status": "processing",
                    }
                    result = mongo_collection.insert_one(metadata)
                    video_doc_id = result.inserted_id
                    
                    st.session_state.processed_files.add(file_id)
                    
                    # --- Start timing the process ---
                    start_time = time.time()

                    # Process the video from the temp file
                    process_video_and_generate_embeddings(processing_path, video_doc_id, uploaded_file.name, minio_key, status_update, whisper_model)
                    
                    # --- End timing and save duration ---
                    end_time = time.time()
                    duration = end_time - start_time
                    mongo_collection.update_one(
                        {"_id": video_doc_id},
                        {"$set": {"processing_duration_seconds": duration}}
                    )

                    st.rerun()

                except Exception as e:
                    status_placeholder.error(f"An error occurred: {e}")
                    if 'video_doc_id' in locals() and 'video_doc_id' in locals() and video_doc_id:
                        mongo_collection.update_one(
                            {"_id": video_doc_id},
                            {"$set": {"processing_status": "failed", "error_message": str(e)}}
                        )
                finally:
                    # Clean up the temporary file
                    if 'processing_path' in locals() and os.path.exists(processing_path):
                        os.remove(processing_path)
                    elif os.path.exists(temp_video_path):
                        os.remove(temp_video_path)

    st.divider()
    st.header("Processed Videos")
    
    videos = list(mongo_collection.find().sort("upload_timestamp", -1))
    if not videos:
        st.info("No videos have been processed yet.")
    else:
        for video in videos:
            status = video.get('processing_status', 'N/A')
            title = f"{video['original_filename']} (Status: {status})"
            
            if status == 'completed' and 'processing_duration_seconds' in video:
                duration_str = format_duration(video.get('processing_duration_seconds'))
                if duration_str != "N/A":
                    title += f" - Processed in {duration_str}"

            with st.expander(title):
                st.json(
                    {k: (v.isoformat() if isinstance(v, datetime) else str(v)) for k, v in video.items()}
                )

# --- Search Tab ---
with tab2:
    st.header("Search Within Videos")

    @retry_on_network_error(max_retries=3, delay=1.0)
    def query_index(index, query_vec, top_k_val=8):
        return index.query(vector=query_vec, top_k=top_k_val, include_metadata=True)

    st.markdown("""
        <style>
        .search-container .stTextArea textarea {
            border-radius: 10px;
            min-height: 80px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.info("You can search with text, an image, or an audio clip. Uploading a file will take priority over text search.")
    
    search_query = st.text_area(
        "Search by Text",
        placeholder="Describe a scene, an action, or a quote from the videos...",
    )
    
    uploaded_search_file = st.file_uploader(
        "Search by Image or Audio",
        type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a'],
        key="search_uploader"
    )

    if search_query or uploaded_search_file:
        
        sorted_results = []
        search_tasks = {}

        if uploaded_search_file:
            file_type = uploaded_search_file.type
            st.info(f"Performing search using uploaded {file_type.split('/')[0]}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_search_file.name)[1]) as temp_file:
                temp_file.write(uploaded_search_file.getvalue())
                temp_file_path = temp_file.name
            
            try:
                if 'audio' in file_type:
                    with st.spinner("Searching for matching audio segments..."):
                        sorted_results = perform_audio_search(temp_file_path)
                
                elif 'image' in file_type:
                    with st.spinner("Processing uploaded image..."):
                        pil_image = Image.open(temp_file_path)
                        query_embedding_tensor, _, _ = get_batch_embeddings(pil_images=[pil_image], audio_paths=[], texts=[], device=device, model=model)
                        if query_embedding_tensor.nelement() > 0:
                            search_tasks[video_pinecone_index] = query_embedding_tensor[0].cpu().numpy().tolist()
                        
                        desc = generate_scene_descriptions([pil_image], caption_processor, caption_model, device)[0]
                        if desc:
                            _, _, desc_embedding_tensor = get_batch_embeddings(pil_images=[], audio_paths=[], texts=[desc], device=device, model=model)
                            if desc_embedding_tensor.nelement() > 0:
                                search_tasks[desc_pinecone_index] = desc_embedding_tensor[0].cpu().numpy().tolist()

            finally:
                os.remove(temp_file_path)

        else: # Text search
            st.info("Performing search using text query...")
            with st.spinner("Generating embedding for your text query..."):
                _, _, query_embedding_tensor = get_batch_embeddings(pil_images=[], audio_paths=[], texts=[search_query], device=device, model=model)
                if query_embedding_tensor.nelement() > 0:
                    query_vector = query_embedding_tensor[0].cpu().numpy().tolist()
                    search_tasks[video_pinecone_index] = query_vector
                    search_tasks[audio_pinecone_index] = query_vector
                    search_tasks[text_pinecone_index] = query_vector
                    search_tasks[desc_pinecone_index] = query_vector
        
        # --- This block now handles TEXT and IMAGE searches ---
        if search_tasks:
            with st.spinner("Searching for matching video segments..."):

                results_map = {}
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_index = {executor.submit(query_index, index, vec): index for index, vec in search_tasks.items()}
                    try:
                        for future in concurrent.futures.as_completed(future_to_index):
                            index = future_to_index[future]
                            results_map[index] = future.result(timeout=30)
                    except concurrent.futures.TimeoutError:
                        st.error("Search query timed out. Please try again.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                    else:
                        # (The rest of the original search logic for merging and scoring)
                        merged_results = {} 
                        
                        def process_results(results, source_name):
                            for match in results.matches:
                                scene_uuid = match.metadata.get("scene_uuid")
                                if not scene_uuid: continue
                                
                                if scene_uuid not in merged_results:
                                    merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
                                else:
                                    merged_results[scene_uuid]['scores'][source_name] = match.score
                                    if match.score > merged_results[scene_uuid]['match'].score:
                                        merged_results[scene_uuid]['match'] = match

                        index_to_name = {
                            video_pinecone_index: 'video',
                            audio_pinecone_index: 'audio',
                            text_pinecone_index: 'text',
                            desc_pinecone_index: 'desc'
                        }
                        
                        for index, results in results_map.items():
                            source_name = index_to_name.get(index)
                            if source_name and results:
                                process_results(results, source_name)
                        
                        final_results = []
                        for item in merged_results.values():
                            best_match = item['match']
                            scores_dict = item['scores']
                            final_score = compute_final_score(scores_dict)
                            best_match.score = final_score
                            source_details = [source for source, score in sorted(scores_dict.items())]
                            best_match.metadata['source'] = ', '.join(source_details)
                            final_results.append(best_match)
                        
                        sorted_results = sorted(final_results, key=lambda x: x.score, reverse=True)

        # --- Final rendering for ALL search types ---
        if sorted_results:
            merged_clips = merge_overlapping_clips(sorted_results)
            render_search_results_grid(merged_clips, original_count=len(sorted_results))
        else:
            st.warning("No matching scenes found.")

        # st.success(f"Found {len(sorted_results)} relevant results.")
        # st.info(f"Found {len(sorted_results)} relevant results (merged from {original_count} results).")

        st.markdown("---")

# --- Analyze Tab ---
with tab3:
    st.header("Analyse Your Video")

    def get_query_intent(prompt: str) -> str:
        """Classifies the user's prompt as 'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST' using an LLM."""
        if not openai_client:
            return "SPECIFIC_QUESTION"  # Default fallback if OpenAI isn't configured

        system_prompt = """You are an intent detection agent. Classify the user's query about a video transcript as either 'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST'.
- SPECIFIC_QUESTION is for queries asking about a particular detail, fact, or event.
- HOLISTIC_REQUEST is for queries asking for a summary, highlights, or an overview of the entire video.
Your response must be ONLY 'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST'."""

        try:
            # This is a quick, cheap call, so we don't need the full retry decorator
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=10,
                timeout=15.0
            )
            intent = response.choices[0].message.content.strip()
            # Validate the response from the LLM
            if intent in ["SPECIFIC_QUESTION", "HOLISTIC_REQUEST"]:
                return intent
            return "SPECIFIC_QUESTION"  # Fallback on any unexpected response
        except Exception:
            return "SPECIFIC_QUESTION"  # Fallback on API error

    def get_llm_analysis_json(prompt: str, context: str, video_duration: float) -> dict | None:
        """
        Sends a prompt and context to the LLM and requests a structured JSON response.
        """
        if not openai_client:
            st.error("OpenAI client is not configured. Please add your API key to secrets.")
            return None

        system_prompt = f"""You are a video content analyst. Your goal is to identify a diverse set of distinct moments from different parts of the video based on the user's request.
The total duration of the video is {int(video_duration)} seconds.

You will receive a video transcript context, with each line prefixed by its timestamp (e.g., [start-end s]). Provide a comprehensive, detailed response based *only* on the provided context. If the answer is not in the context, state that clearly.

Your response MUST be a JSON object with two keys:
1. "analysis_text": A string containing your full, detailed analysis.
2. "clips": A list of JSON objects representing mentioned clips. Each object must have "description" (string) and "start_time" (integer in seconds).

IMPORTANT RULES for the "clips" list:
- Each clip in the 'clips' list MUST have a unique 'start_time'.
- The 'start_time' for any clip MUST NOT exceed the video duration of {int(video_duration)} seconds.
- Base your 'start_time' on the timestamps provided in the context (e.g., [15.5-20.0s]).
- If multiple relevant events occur at the same start time, consolidate them into a single, more descriptive clip entry.
- If no specific clips are mentioned or found, provide an empty list.

Example clip: {{"description": "The speaker discusses Q3 earnings and future projections.", "start_time": 125}}"""

        user_message = f"""Context from Video:
---
{context}
---

User Request: {prompt}

Please provide your analysis in the required JSON format based on the context above."""

        try:
            @retry_on_network_error(max_retries=3, delay=2.0)
            def call_openai_api():
                return openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                    timeout=60.0
                )

            with st.spinner("Analyzing with AI..."):
                response = call_openai_api()
                analysis_result = response.choices[0].message.content
                return json.loads(analysis_result)
        except Exception as e:
            st.error(f"OpenAI analysis failed: {e}")
            return None

    # Initialize session state
    if 'analyze_prompt' not in st.session_state:
        st.session_state.analyze_prompt = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    processed_videos = list(mongo_collection.find(
        {"processing_status": "completed"},
        {"original_filename": 1, "_id": 1, "minio_key": 1, "transcript_minio_key": 1, "video_duration": 1}
    ).sort("upload_timestamp", -1))

    if not processed_videos:
        st.info("No videos have been processed yet. Please upload a video first.")
    else:
        video_options = {v['original_filename']: v for v in processed_videos}
        
        # Reset analysis if video changes
        if 'last_analyzed_video' not in st.session_state:
            st.session_state.last_analyzed_video = None

        selected_video_name = st.selectbox(
            "Select a video to analyze",
            options=list(video_options.keys())
        )

        if st.session_state.last_analyzed_video != selected_video_name:
            st.session_state.analysis_result = None
            st.session_state.last_analyzed_video = selected_video_name

        if selected_video_name:
            col1, col2 = st.columns([1, 1])
            selected_video_doc = video_options[selected_video_name]

            with col1:
                video_bytes = get_video_bytes(selected_video_doc)
                if video_bytes:
                    st.video(video_bytes)
                else:
                    st.error("Could not load video for playback.")

                st.text_area("Your analysis prompt:", key="analyze_prompt", height=150)

                if st.button("🔍 Analyze Video", use_container_width=True):
                    if st.session_state.analyze_prompt:
                        prompt = st.session_state.analyze_prompt
                        result = None
                        video_duration = selected_video_doc.get("video_duration", 0.0)

                        # --- NEW LLM-BASED INTENT ROUTER ---
                        with st.spinner("Analyzing your question..."):
                            intent = get_query_intent(prompt)
                        
                        print(f"Detected query intent: {intent}")
                        # st.info(f"Query classified as: **{intent.replace('_', ' ').title()}**")

                        if intent == "HOLISTIC_REQUEST":
                            # HOLISTIC PATH
                            with st.spinner("Fetching full transcript for holistic analysis..."):
                                transcript_segments = get_transcript_segments(selected_video_doc)
                                if transcript_segments:
                                    context_parts = [f"[{seg.get('start', 0.0):.1f}-{seg.get('end', 0.0):.1f}s] {seg.get('text', '').strip()}" for seg in transcript_segments]
                                    context = "\n".join(context_parts)
                                    result = get_llm_analysis_json(prompt, context, video_duration)
                                else:
                                    st.error("Could not retrieve transcript to perform analysis.")
                        else: # intent == "SPECIFIC_QUESTION"
                            # RAG PATH
                            with st.spinner("Searching for relevant video segments..."):
                                _, _, query_embedding = get_batch_embeddings(pil_images=[], audio_paths=[], texts=[prompt], device=device, model=model)
                                if query_embedding.nelement() > 0:
                                    query_vector = query_embedding[0].cpu().numpy().tolist()
                                    
                                    # Search both text and description indexes, filtering by the specific video
                                    search_results = []
                                    text_res = text_pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"video_doc_id": str(selected_video_doc["_id"])})
                                    desc_res = desc_pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True, filter={"video_doc_id": str(selected_video_doc["_id"])})
                                    
                                    if text_res.matches: search_results.extend(text_res.matches)
                                    if desc_res.matches: search_results.extend(desc_res.matches)
                                    
                                    if search_results:
                                        unique_clips_context = {}
                                        for match in sorted(search_results, key=lambda x: x.score, reverse=True):
                                            meta = match.metadata
                                            start = meta.get('start_time', 0.0)
                                            end = meta.get('end_time', 0.0)
                                            text = meta.get("transcript") or meta.get("description")
                                            if text:
                                                # Use int(start) as key to avoid duplicates from very close timestamps
                                                if int(start) not in unique_clips_context:
                                                    unique_clips_context[int(start)] = f"[{start:.1f}-{end:.1f}s] {text}"

                                        context = "\n".join(unique_clips_context.values())
                                        result = get_llm_analysis_json(prompt, context, video_duration)
                                    else:
                                        st.warning("Couldn't find relevant segments for this query. Trying with full transcript...")
                                        transcript_segments = get_transcript_segments(selected_video_doc)
                                        if transcript_segments:
                                            context_parts = [f"[{seg.get('start', 0.0):.1f}-{seg.get('end', 0.0):.1f}s] {seg.get('text', '').strip()}" for seg in transcript_segments]
                                            context = "\n".join(context_parts)
                                            result = get_llm_analysis_json(prompt, context, video_duration)
                                        else:
                                            st.error("Could not retrieve transcript to perform analysis.")
                                else:
                                    st.error("Could not generate embedding for your query.")
                        
                        st.session_state.analysis_result = result
                    else:
                        st.warning("Please enter a prompt to start the analysis.")
                
                if st.session_state.analysis_result:
                    st.markdown("---")
                    st.subheader("📊 Analysis Result")
                    analysis_data = st.session_state.analysis_result
                    st.markdown(analysis_data.get("analysis_text", "No analysis text found."))
                    
                    clips = analysis_data.get("clips", [])
                    if clips and video_bytes:
                        st.markdown("---")
                        st.subheader("🎬 Playable Clips")
                        for i, clip in enumerate(clips):
                            start_time = clip.get("start_time", 0)
                            if isinstance(start_time, (int, float)):
                                clip_start_ts = time.strftime('%H:%M:%S', time.gmtime(start_time))
                                if st.button(f"{clip.get('description', 'Untitled Clip')} (at {clip_start_ts})", key=f"clip_{i}_{start_time}"):
                                    # This part requires re-running to update the video player, which is complex in Streamlit.
                                    # For now, we show individual clips.
                                    pass
                                st.video(video_bytes, start_time=int(start_time))

            with col2:
                st.subheader("Suggested Prompts")
                prompts = [
                    "Generate hashtags and topics",
                    "Summarize this video",
                    "What are highlighted moments of this video?",
                    "Chapterize this video",
                    "Classify this video based on Youtube categories. Output as JSON format.",
                    "Which audience is the video suitable for, and why?",
                    "Break down the video by main event and timestamp"
                ]

                def set_prompt(p):
                    st.session_state.analyze_prompt = p

                # Display first two prompts, which are always visible
                for p in prompts[:2]:
                    st.button(p, key=f"prompt_{p}", on_click=set_prompt, args=(p,), use_container_width=True)

                # Use an expander for the rest of the prompts
                with st.expander("See more"):
                    for p in prompts[2:]:
                        st.button(p, key=f"prompt_{p}", on_click=set_prompt, args=(p,), use_container_width=True)

# --- Reset Functionality ---
st.sidebar.title("⚠️ Danger Zone")

def empty_minio_bucket(bucket_name):
    try:
        objects_to_delete = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in objects_to_delete:
            delete_keys = {'Objects': [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]}
            if delete_keys['Objects']:
                 s3_client.delete_objects(Bucket=bucket_name, Delete=delete_keys)
    except Exception as e:
        st.sidebar.warning(f"Warning: Could not empty bucket {bucket_name}: {e}")

if st.sidebar.button("Reset All Data"):
    st.session_state.show_confirmation = True

if 'show_confirmation' in st.session_state and st.session_state.show_confirmation:
    st.sidebar.warning("This will delete all data from MinIO, MongoDB, and Pinecone. Are you sure?")
    if st.sidebar.button("Yes, I am sure"):
        with st.spinner("Deleting all data..."):
            # 1. Delete all from Pinecone
            try:
                video_pinecone_index.delete(delete_all=True, namespace="__default__")
                audio_pinecone_index.delete(delete_all=True, namespace="__default__")
                text_pinecone_index.delete(delete_all=True, namespace="__default__")
                desc_pinecone_index.delete(delete_all=True, namespace="__default__")
            except Exception as e:
                # Handle case where index is empty or namespace doesn't exist
                # This is fine - it means there's nothing to delete
                if "NotFound" in str(type(e).__name__) or "namespace not found" in str(e).lower():
                    st.sidebar.info("Pinecone index is already empty or namespace doesn't exist.")
                else:
                    st.sidebar.warning(f"Warning: Could not delete from Pinecone: {e}")
            
            # 2. Delete all from MinIO
            empty_minio_bucket(config["minio"]["bucket_name"])
            empty_minio_bucket(config["minio"]["transcripts_bucket_name"])
            
            # 3. Delete all from MongoDB
            try:
                mongo_collection.delete_many({})
            except Exception as e:
                st.sidebar.warning(f"Warning: Could not delete from MongoDB: {e}")

        st.success("All data has been reset.")
        st.session_state.show_confirmation = False
        st.rerun()

    if st.sidebar.button("Cancel"):
        st.session_state.show_confirmation = False
        st.rerun()
