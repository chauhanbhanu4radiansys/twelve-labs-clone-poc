"""
Audio Language Identification using ECAPA-TDNN
Extracts audio from video using ffmpeg (WAV format) and identifies language
using speechbrain/lang-id-voxlingua107-ecapa model.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional

# Monkey patch for torchaudio 2.9.0+ compatibility with SpeechBrain
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def list_audio_backends():
            return ['soundfile', 'sox']
        torchaudio.list_audio_backends = list_audio_backends
except ImportError:
    pass

try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    raise ImportError(
        "Please install speechbrain: pip install speechbrain\n"
        "Also ensure torch and torchaudio are installed: pip install torch torchaudio"
    )


def extract_audio_ffmpeg(video_path: str, output_wav_path: Optional[str] = None) -> str:
    """
    Extracts audio from video file using ffmpeg and saves as WAV format.
    
    Args:
        video_path: Path to the input video file
        output_wav_path: Optional path for output WAV file. If None, creates temp file.
    
    Returns:
        Path to the extracted WAV audio file
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffmpeg fails or is not installed
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_wav_path is None:
        # Create temporary WAV file
        temp_dir = tempfile.gettempdir()
        video_name = Path(video_path).stem
        output_wav_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    
    # Extract audio using ffmpeg
    # -i: input file
    # -vn: disable video
    # -acodec pcm_s16le: PCM 16-bit little-endian audio codec (WAV format)
    # -ar 16000: sample rate 16kHz (required for ECAPA-TDNN)
    # -ac 1: mono channel
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",  # Overwrite output file if exists
        "-loglevel", "error",  # Suppress ffmpeg output
        output_wav_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Audio extracted to WAV: {output_wav_path}")
        return output_wav_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install ffmpeg:\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  - Windows: Download from https://ffmpeg.org/download.html"
        )


def load_lid_model(model_dir: Optional[str] = None):
    """
    Loads the ECAPA-TDNN language identification model from SpeechBrain.
    Model: speechbrain/lang-id-voxlingua107-ecapa (supports 107 languages)
    
    Args:
        model_dir: Optional directory to save/load the model. Defaults to pretrained_models/lang-id-voxlingua107-ecapa
    
    Returns:
        Loaded EncoderClassifier model
    """
    if model_dir is None:
        model_dir = "pretrained_models/lang-id-voxlingua107-ecapa"
    
    print("Loading ECAPA-TDNN language identification model...")
    try:
        language_id = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir=model_dir
        )
        print("✓ ECAPA-TDNN model loaded successfully")
        return language_id
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ECAPA-TDNN model: {e}\n"
            "Make sure you have internet connection for first-time model download."
        )


def get_lid_from_audio(audio_wav_path: str, model) -> Tuple[str, float]:
    """
    Gets language identification (LID) from a WAV audio file using ECAPA-TDNN model.
    
    Args:
        audio_wav_path: Path to the WAV audio file
        model: Loaded ECAPA-TDNN EncoderClassifier model
    
    Returns:
        Tuple of (language_code, confidence_score)
        - language_code: ISO 639-1 language code (e.g., 'en', 'hi', 'es', 'fr')
        - confidence_score: Confidence score between 0.0 and 1.0
    
    Raises:
        RuntimeError: If language identification fails
    """
    if not os.path.exists(audio_wav_path):
        raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")
    
    print(f"Identifying language from audio: {audio_wav_path}")
    
    try:
        # Load the audio file
        signal = model.load_audio(audio_wav_path)
        
        # Classify the language
        prediction = model.classify_batch(signal)
        
        # Extract the language code and confidence score
        # prediction format: (logits, posterior, language_code, confidence)
        language_code = prediction[3][0]  # Language code (e.g., 'en', 'hi', 'es')
        confidence = prediction[1].exp().item()  # Confidence score (0-1)
        
        print(f"✓ Language identified: {language_code.upper()} (confidence: {confidence:.2%})")
        return language_code, confidence
        
    except Exception as e:
        raise RuntimeError(f"Error during language identification: {e}")


def get_lid_from_video(video_path: str, keep_audio: bool = False, output_wav_path: Optional[str] = None) -> dict:
    """
    Complete pipeline: Extract audio from video (WAV) -> Get LID using ECAPA-TDNN.
    
    Args:
        video_path: Path to the input video file
        keep_audio: If True, keeps the extracted WAV file. If False, deletes it after processing.
        output_wav_path: Optional path for output WAV file. If None, creates temp file.
    
    Returns:
        Dictionary containing:
        - video_path: Path to input video
        - audio_path: Path to extracted WAV file
        - language_code: Detected language code
        - confidence: Confidence score (0-1)
    """
    print(f"\n{'='*60}")
    print(f"Processing video for language identification")
    print(f"{'='*60}")
    print(f"Video: {video_path}\n")
    
    temp_audio_path = None
    try:
        # Step 1: Extract audio from video using ffmpeg (WAV format)
        temp_audio_path = extract_audio_ffmpeg(video_path, output_wav_path)
        
        # Step 2: Load ECAPA-TDNN model
        lid_model = load_lid_model()
        
        # Step 3: Get language identification
        language_code, confidence = get_lid_from_audio(temp_audio_path, lid_model)
        
        # Prepare result
        result = {
            "video_path": video_path,
            "audio_path": temp_audio_path,
            "language_code": language_code,
            "confidence": confidence
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("LANGUAGE IDENTIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Video File: {video_path}")
        print(f"Audio File (WAV): {temp_audio_path}")
        print(f"Detected Language: {language_code.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*60}\n")
        
        return result
        
    finally:
        # Clean up temporary audio file if requested
        if not keep_audio and temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                print(f"Cleaned up temporary audio file: {temp_audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_audio_path}: {e}")


def main():
    """
    Main entry point for command-line usage.
    """
    if len(sys.argv) < 2:
        print("Usage: python audio_lid_ecapa.py <video_file_path> [output_wav_path]")
        print("\nExample:")
        print("  python audio_lid_ecapa.py /path/to/video.mp4")
        print("  python audio_lid_ecapa.py /path/to/video.mp4 /path/to/output.wav")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_wav_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert to absolute path
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(video_path)
    
    try:
        result = get_lid_from_video(video_path, keep_audio=(output_wav_path is not None), output_wav_path=output_wav_path)
        print(f"\n✓ Processing completed successfully!")
        print(f"  Language: {result['language_code'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

