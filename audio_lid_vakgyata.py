"""
Audio Language Identification using Vakgyata Base Model
Extracts audio from video using ffmpeg (WAV format) and identifies language
using onecxi/vakgyata-base model (designed for Indian languages).
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
except ImportError:
    raise ImportError(
        "Please install required packages:\n"
        "pip install torch torchaudio transformers"
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
    # -ar 16000: sample rate 16kHz (required for the model)
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


def load_lid_model(model_name: str = "onecxi/vakgyata-base", model_dir: Optional[str] = None):
    """
    Loads the Vakgyata Base language identification model from Hugging Face.
    Model: onecxi/vakgyata-base (designed for Indian languages)
    
    Args:
        model_name: Hugging Face model identifier
        model_dir: Optional directory to save/load the model
    
    Returns:
        Tuple of (model, processor, device)
    """
    print(f"Loading Vakgyata Base LID model: {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load the feature extractor (processor)
        processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=model_dir)
        
        # Load the model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=model_dir
        )
        model.to(device)
        model.eval()
        
        print("✓ Vakgyata Base LID model loaded successfully")
        return model, processor, device
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Vakgyata Base LID model: {e}\n"
            "Make sure you have internet connection for first-time model download."
        )


def get_lid_from_audio(audio_wav_path: str, model, processor, device) -> Tuple[str, float, int]:
    """
    Gets language identification (LID) from a WAV audio file using Vakgyata Base model.
    
    Args:
        audio_wav_path: Path to the WAV audio file
        model: Loaded Vakgyata Base model
        processor: Audio feature extractor
        device: Torch device (CPU or CUDA)
    
    Returns:
        Tuple of (language_code, confidence_score, language_index)
        - language_code: Language code or label
        - confidence_score: Confidence score between 0.0 and 1.0
        - language_index: Predicted language index
    
    Raises:
        RuntimeError: If language identification fails
    """
    if not os.path.exists(audio_wav_path):
        raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")
    
    print(f"Identifying language from audio: {audio_wav_path}")
    
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_wav_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Squeeze to remove channel dimension for processor
        waveform = waveform.squeeze()
        
        # Process audio with feature extractor
        inputs = processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get logits
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted language index
            language_index = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, language_index].item()
        
        # Get language label from model config
        if hasattr(model.config, 'id2label') and model.config.id2label:
            language_code = model.config.id2label[language_index]
        elif hasattr(model.config, 'label2id') and model.config.label2id:
            # Reverse lookup
            label2id = model.config.label2id
            language_code = [k for k, v in label2id.items() if v == language_index][0] if label2id else str(language_index)
        else:
            # Fallback: use index as label
            language_code = str(language_index)
        
        print(f"✓ Language identified: {language_code} (confidence: {confidence:.2%}, index: {language_index})")
        return language_code, confidence, language_index
        
    except Exception as e:
        raise RuntimeError(f"Error during language identification: {e}")


def get_lid_from_video(video_path: str, keep_audio: bool = False, output_wav_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete pipeline: Extract audio from video (WAV) -> Get LID using Vakgyata Base model.
    
    Args:
        video_path: Path to the input video file
        keep_audio: If True, keeps the extracted WAV file. If False, deletes it after processing.
        output_wav_path: Optional path for output WAV file. If None, creates temp file.
    
    Returns:
        Dictionary containing:
        - video_path: Path to input video
        - audio_path: Path to extracted WAV file
        - language_code: Detected language code/label
        - confidence: Confidence score (0-1)
        - language_index: Predicted language index
    """
    print(f"\n{'='*60}")
    print(f"Processing video for language identification")
    print(f"{'='*60}")
    print(f"Video: {video_path}\n")
    
    temp_audio_path = None
    try:
        # Step 1: Extract audio from video using ffmpeg (WAV format)
        temp_audio_path = extract_audio_ffmpeg(video_path, output_wav_path)
        
        # Step 2: Load Vakgyata Base model
        lid_model, processor, device = load_lid_model()
        
        # Step 3: Get language identification
        language_code, confidence, language_index = get_lid_from_audio(
            temp_audio_path, lid_model, processor, device
        )
        
        # Prepare result
        result = {
            "video_path": video_path,
            "audio_path": temp_audio_path,
            "language_code": language_code,
            "confidence": confidence,
            "language_index": language_index
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("LANGUAGE IDENTIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Video File: {video_path}")
        print(f"Audio File (WAV): {temp_audio_path}")
        print(f"Detected Language: {language_code}")
        print(f"Language Index: {language_index}")
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
        print("Usage: python audio_lid_vakgyata.py <video_file_path> [output_wav_path]")
        print("\nExample:")
        print("  python audio_lid_vakgyata.py /path/to/video.mp4")
        print("  python audio_lid_vakgyata.py /path/to/video.mp4 /path/to/output.wav")
        print("\nModel: onecxi/vakgyata-base (designed for Indian languages)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_wav_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert to absolute path
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(video_path)
    
    try:
        result = get_lid_from_video(
            video_path,
            keep_audio=(output_wav_path is not None),
            output_wav_path=output_wav_path
        )
        print(f"\n✓ Processing completed successfully!")
        print(f"  Language: {result['language_code']}")
        print(f"  Language Index: {result['language_index']}")
        print(f"  Confidence: {result['confidence']:.2%}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
