"""
Video Language Identification using ECAPA-TDNN
Takes a video file path as input, extracts audio using ffmpeg, 
and identifies the language using ECAPA-TDNN model.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Monkey patch for torchaudio 2.9.0+ compatibility with SpeechBrain
# The list_audio_backends() method was removed in newer torchaudio versions
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        # Add a dummy implementation for compatibility
        def list_audio_backends():
            # Return a list with common backends for compatibility
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


def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    """
    Extracts audio from video file using ffmpeg and converts to WAV format.
    
    Args:
        video_path: Full path to the input video file
        output_audio_path: Optional path for output WAV file. If None, creates temp file.
    
    Returns:
        Path to the extracted WAV audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_audio_path is None:
        # Create temporary WAV file
        temp_dir = tempfile.gettempdir()
        video_name = Path(video_path).stem
        output_audio_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    # Extract audio using ffmpeg
    # -i: input file
    # -vn: disable video
    # -acodec pcm_s16le: PCM 16-bit little-endian audio codec
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
        output_audio_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Audio extracted successfully to: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install ffmpeg:\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  - Windows: Download from https://ffmpeg.org/download.html"
        )


def load_ecapa_lid_model():
    """
    Loads the ECAPA-TDNN language identification model from SpeechBrain.
    This model can identify 107 languages.
    
    Returns:
        Loaded EncoderClassifier model
    """
    print("Loading ECAPA-TDNN language identification model...")
    try:
        # Load the pre-trained ECAPA-TDNN model for language identification
        # This model is trained on VoxLingua107 dataset and supports 107 languages
        language_id = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang-id-voxlingua107-ecapa"
        )
        print("✓ ECAPA-TDNN model loaded successfully")
        return language_id
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ECAPA-TDNN model: {e}\n"
            "Make sure you have internet connection for first-time model download."
        )


def identify_language_with_ecapa(audio_path: str, model) -> tuple:
    """
    Identifies language using ECAPA-TDNN model.
    
    Args:
        audio_path: Path to the audio WAV file
        model: Loaded ECAPA-TDNN EncoderClassifier model
    
    Returns:
        Tuple of (language_code, confidence_score)
    """
    print(f"Identifying language for audio: {audio_path}")
    
    try:
        # Load the audio file
        signal = model.load_audio(audio_path)
        
        # Classify the language
        prediction = model.classify_batch(signal)
        
        # Extract the language code and confidence score
        # prediction format: (logits, posterior, language_code, confidence)
        language_code = prediction[3][0]  # Language code (e.g., 'en', 'hi', 'es')
        confidence = prediction[1].exp().item()  # Confidence score (0-1)
        
        print(f"✓ Language identification completed")
        return language_code, confidence
        
    except Exception as e:
        raise RuntimeError(f"Error during language identification: {e}")


def process_video_for_lid(video_path: str) -> dict:
    """
    Main pipeline: Video -> Audio (WAV) -> ECAPA-TDNN LID -> Print Language
    
    Args:
        video_path: Full path to the input video file
    
    Returns:
        Dictionary containing language identification results
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"{'='*60}\n")
    
    temp_audio_path = None
    try:
        # Step 1: Extract audio from video
        temp_audio_path = extract_audio_from_video(video_path)
        
        # Step 2: Load ECAPA-TDNN model
        lid_model = load_ecapa_lid_model()
        
        # Step 3: Identify language
        language_code, confidence = identify_language_with_ecapa(
            temp_audio_path,
            lid_model
        )
        
        # Prepare result
        result = {
            "video_path": video_path,
            "audio_path": temp_audio_path,
            "language_code": language_code,
            "confidence": confidence
        }
        
        # Step 4: Print results on screen
        print(f"\n{'='*60}")
        print("LANGUAGE IDENTIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Video File: {video_path}")
        print(f"Audio File: {temp_audio_path}")
        print(f"Detected Language: {language_code.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*60}\n")
        
        return result
        
    finally:
        # Clean up temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
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
        print("Usage: python video_lid_ecapa.py <video_file_path>")
        print("\nExample:")
        print("  python video_lid_ecapa.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Convert to absolute path
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(video_path)
    
    try:
        result = process_video_for_lid(video_path)
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

