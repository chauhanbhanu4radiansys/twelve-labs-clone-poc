"""
Video processing functions: transcription, scene detection, frame extraction
"""
import os
import sys
import subprocess
import tempfile
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from contextlib import redirect_stderr
from pydub import AudioSegment

from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector


def extract_audio_from_video(video_path: str) -> Optional[str]:
    """
    Extracts audio from a video and saves it as a WAV file.
    Returns the path to the WAV file.
    """
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


def find_segment_by_scene_end(scene_end_time: float, segments: List[Dict], tolerance: float = 0.5) -> Optional[Dict]:
    """
    Finds the transcript segment whose 'end' time matches the scene's end time.
    """
    if not segments:
        return None
    
    best_match = None
    min_diff = float('inf')
    
    for segment in segments:
        segment_end = segment.get('end', 0)
        diff = abs(scene_end_time - segment_end)
        adaptive_tolerance = min(tolerance, max(0.1, scene_end_time * 0.1))
        
        if diff < adaptive_tolerance and diff < min_diff:
            min_diff = diff
            best_match = segment
    
    if best_match is None:
        for segment in segments:
            segment_end = segment.get('end', 0)
            if segment_end <= scene_end_time:
                diff = scene_end_time - segment_end
                if diff < min_diff:
                    min_diff = diff
                    best_match = segment
    
    if best_match is None:
        for segment in segments:
            segment_end = segment.get('end', 0)
            diff = abs(scene_end_time - segment_end)
            if diff < min_diff:
                min_diff = diff
                best_match = segment
    
    return best_match


def detect_scenes_with_adaptive_detector(video_path: str) -> List:
    """
    Detect scenes using AdaptiveDetector.
    Returns all detected scenes without any minimum duration filtering.
    """
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
    """Get video duration in seconds using multiple methods."""
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
    """
    Prepares all necessary data for a single scene for batch processing.
    """
    chunk_path, scene, fps, chunk_start_offset, video_name, video_id, global_scene_index, transcript_segments, full_audio_segment = args
    
    abs_start_time = chunk_start_offset + scene[0].get_seconds()
    abs_end_time = chunk_start_offset + scene[1].get_seconds()

    # Frame Extraction
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
    
    # Audio Slicing
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
    
    return {
        "pil_image": resize_frame_optimized(frame_np),
        "audio_path": scene_audio_path,
        "transcript": find_segment_by_scene_end(abs_end_time, transcript_segments).get('text', '').strip() if find_segment_by_scene_end(abs_end_time, transcript_segments) else '',
        "video_name": video_name,
        "video_id": video_id,
        "scene_global_index": global_scene_index,
        "abs_start_time": abs_start_time,
        "abs_end_time": abs_end_time,
    }

