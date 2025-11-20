"""
Video Language Identification and Transcription Pipeline
Uses ECAPA-TDNN for language detection, classifies as Indian/non-Indian,
then uses Indic Seamless for Indian languages or Whisper for others.

Indic Seamless: https://model.aibase.com/models/details/1915693340919750658
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

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

try:
    import whisper
except ImportError:
    raise ImportError(
        "Please install openai-whisper:\n"
        "pip install openai-whisper"
    )

try:
    import torch
    from transformers import SeamlessM4Tv2ForSpeechToText
    from transformers import SeamlessM4TTokenizer, SeamlessM4TFeatureExtractor
except ImportError:
    raise ImportError(
        "Please install transformers:\n"
        "pip install transformers"
    )

# Indian languages (all Indian languages, including those not supported by Indic Seamless)
INDIAN_LANGUAGES_ALL = {
    "as",  # Assamese
    "bn",  # Bengali
    "gu",  # Gujarati
    "hi",  # Hindi
    "ta",  # Tamil
    "te",  # Telugu
    "ur",  # Urdu
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "sd",  # Sindhi
    "ne",  # Nepali
    "pa",  # Punjabi (Indian but not in Indic Seamless)
    "or",  # Odia/Oriya (Indian but not in Indic Seamless)
    "en",  # English (also supported)
}

# Indian languages supported by Indic Seamless
# Languages: en, as, bn, gu, hi, ta, te, ur, kn, ml, mr, sd, ne
INDIAN_LANGUAGES_INDIC_SEAMLESS = {
    "as",  # Assamese
    "bn",  # Bengali
    "gu",  # Gujarati
    "hi",  # Hindi
    "ta",  # Tamil
    "te",  # Telugu
    "ur",  # Urdu
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "sd",  # Sindhi
    "ne",  # Nepali
    "en",  # English (also supported)
}

# Mapping from ISO codes to Indic Seamless target language codes
INDIC_SEAMLESS_TGT_LANG_MAP = {
    "as": "asm",  # Assamese
    "bn": "ben",  # Bengali
    "gu": "guj",  # Gujarati
    "hi": "hin",  # Hindi
    "ta": "tam",  # Tamil
    "te": "tel",  # Telugu
    "ur": "urd",  # Urdu
    "kn": "kan",  # Kannada
    "ml": "mal",  # Malayalam
    "mr": "mar",  # Marathi
    "sd": "snd",  # Sindhi
    "ne": "nep",  # Nepali
    "en": "eng",  # English
}


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
    # -ar 16000: sample rate 16kHz (required for both models)
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


def load_ecapa_lid_model(model_dir: Optional[str] = None):
    """
    Loads the ECAPA-TDNN language identification model from SpeechBrain.
    Model: speechbrain/lang-id-voxlingua107-ecapa (supports 107 languages)
    
    Args:
        model_dir: Optional directory to save/load the model
    
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
        - language_code: ISO 639-1 language code (e.g., 'en', 'hi', 'es')
        - confidence_score: Confidence score between 0.0 and 1.0
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
        language_code_raw = prediction[3][0]  # Language code (e.g., 'en', 'hi', 'es', or 'pa: panjabi')
        
        # Clean language code - extract just the ISO code if format is "code: name"
        if ':' in language_code_raw:
            language_code = language_code_raw.split(':')[0].strip().lower()
        else:
            language_code = language_code_raw.strip().lower()
        
        confidence = prediction[1].exp().item()  # Confidence score (0-1)
        
        print(f"✓ Language identified: {language_code.upper()} (confidence: {confidence:.2%})")
        return language_code, confidence
        
    except Exception as e:
        raise RuntimeError(f"Error during language identification: {e}")


def is_indian_language(language_code: str) -> bool:
    """
    Checks if the detected language is an Indian language.
    
    Args:
        language_code: ISO 639-1 language code
    
    Returns:
        True if Indian language, False otherwise
    """
    return language_code.lower() in INDIAN_LANGUAGES_ALL


def is_supported_by_indic_seamless(language_code: str) -> bool:
    """
    Checks if the detected language is supported by Indic Seamless.
    
    Args:
        language_code: ISO 639-1 language code
    
    Returns:
        True if supported by Indic Seamless, False otherwise
    """
    return language_code.lower() in INDIAN_LANGUAGES_INDIC_SEAMLESS


def load_indic_seamless_model():
    """
    Loads the Indic Seamless model for Indian language transcription.
    Model: ai4bharat/indic-seamless
    
    Returns:
        Tuple of (model, processor, tokenizer, device)
    """
    print("Loading Indic Seamless model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless").to(device)
        processor = SeamlessM4TFeatureExtractor.from_pretrained("ai4bharat/indic-seamless")
        tokenizer = SeamlessM4TTokenizer.from_pretrained("ai4bharat/indic-seamless")
        
        print("✓ Indic Seamless model loaded successfully")
        return model, processor, tokenizer, device
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Indic Seamless model: {e}\n"
            "Make sure you have internet connection for first-time model download."
        )


def transcribe_with_indic_seamless(audio_wav_path: str, language_code: str, model, processor, tokenizer, device) -> str:
    """
    Transcribes audio using Indic Seamless model.
    
    Args:
        audio_wav_path: Path to the WAV audio file (16kHz)
        language_code: ISO 639-1 language code
        model: Loaded Indic Seamless model
        processor: SeamlessM4TFeatureExtractor
        tokenizer: SeamlessM4TTokenizer
        device: Torch device
    
    Returns:
        Transcribed text
    """
    print(f"Transcribing with Indic Seamless (language: {language_code})")
    
    try:
        # Load audio and resample to 16kHz if needed
        audio, orig_freq = torchaudio.load(audio_wav_path)
        if orig_freq != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
        
        # Get target language code for Indic Seamless
        tgt_lang = INDIC_SEAMLESS_TGT_LANG_MAP.get(language_code.lower(), "hin")
        
        # Process audio
        audio_inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        
        # Generate transcription
        with torch.no_grad():
            text_out = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
        
        # Decode text
        transcribed_text = tokenizer.decode(
            text_out,
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True
        )
        
        print(f"✓ Transcription completed with Indic Seamless")
        return transcribed_text
        
    except Exception as e:
        raise RuntimeError(f"Error during Indic Seamless transcription: {e}")


def transcribe_with_whisper(audio_path: str, language: Optional[str] = None, model: Optional[Any] = None) -> Dict:
    """
    Transcribes audio using Whisper.
    
    Args:
        audio_path: Path to the audio file
        language: Optional language code for transcription (if None, auto-detect)
        model: Optional pre-loaded Whisper model (for efficiency)
    
    Returns:
        Dictionary containing transcription results with segments
    """
    print(f"Transcribing audio with Whisper (language: {language or 'auto'})")
    
    try:
        # Load Whisper model if not provided
        if model is None:
            model = whisper.load_model("base")
        
        # Transcribe with optional language specification
        if language:
            try:
                result = model.transcribe(audio_path, language=language)
            except ValueError as e:
                # If language is not supported, fall back to auto-detect
                if "Unsupported language" in str(e):
                    print(f"Warning: Language '{language}' not supported by Whisper, using auto-detect")
                    result = model.transcribe(audio_path)
                else:
                    raise
        else:
            result = model.transcribe(audio_path)
        
        print(f"Transcription completed. Found {len(result.get('segments', []))} segments")
        return result
        
    except Exception as e:
        raise RuntimeError(f"Whisper transcription error: {e}")


def process_video_pipeline(video_path: str, output_json_path: Optional[str] = None, keep_audio: bool = False, output_wav_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete pipeline: Extract audio -> LID -> Classify -> Transcribe (Indic Seamless or Whisper) -> Save to JSON.
    
    Args:
        video_path: Path to the input video file
        output_json_path: Optional path for output JSON file. If None, creates file next to video.
        keep_audio: If True, keeps the extracted WAV file. If False, deletes it after processing.
        output_wav_path: Optional path for output WAV file. If None, creates temp file.
    
    Returns:
        Dictionary containing all processing results
    """
    print(f"\n{'='*60}")
    print(f"Processing video: LID -> Classification -> Transcription")
    print(f"{'='*60}")
    print(f"Video: {video_path}\n")
    
    temp_audio_path = None
    indic_seamless_model = None
    indic_seamless_processor = None
    indic_seamless_tokenizer = None
    indic_seamless_device = None
    whisper_model = None
    
    try:
        # Step 1: Extract audio from video using ffmpeg (WAV format)
        temp_audio_path = extract_audio_ffmpeg(video_path, output_wav_path)
        
        # Step 2: Load ECAPA-TDNN model
        lid_model = load_ecapa_lid_model()
        
        # Step 3: Get language identification
        language_code, confidence = get_lid_from_audio(temp_audio_path, lid_model)
        
        # Step 4: Classify as Indian or non-Indian
        is_indian = is_indian_language(language_code)
        print(f"Language classification: {'Indian' if is_indian else 'Non-Indian'}")
        
        # Step 5: Transcribe based on classification
        if is_indian and is_supported_by_indic_seamless(language_code):
            # Use Indic Seamless for Indian languages supported by it
            indic_seamless_model, indic_seamless_processor, indic_seamless_tokenizer, indic_seamless_device = load_indic_seamless_model()
            transcribed_text = transcribe_with_indic_seamless(
                temp_audio_path,
                language_code,
                indic_seamless_model,
                indic_seamless_processor,
                indic_seamless_tokenizer,
                indic_seamless_device
            )
            
            transcription_result = {
                "text": transcribed_text,
                "language": language_code,
                "model": "indic-seamless",
                "segments": []  # Indic Seamless doesn't provide segments by default
            }
        else:
            # Use Whisper for non-Indian languages or Indian languages not supported by Indic Seamless
            if is_indian:
                print(f"Note: {language_code.upper()} is Indian but not supported by Indic Seamless, using Whisper")
            whisper_model = whisper.load_model("base")
            transcription_result = transcribe_with_whisper(
                temp_audio_path,
                language=language_code,
                model=whisper_model
            )
            transcription_result["model"] = "whisper"
        
        # Prepare output
        output_data = {
            "video_path": video_path,
            "audio_path": temp_audio_path,
            "language_detection": {
                "language_code": language_code,
                "confidence": confidence,
                "is_indian": is_indian
            },
            "transcription": {
                "text": transcription_result.get('text', ''),
                "language": transcription_result.get('language', language_code),
                "model": transcription_result.get('model', 'unknown'),
                "segments": transcription_result.get('segments', [])
            }
        }
        
        # Step 6: Write to JSON file
        if output_json_path is None:
            video_dir = os.path.dirname(video_path)
            video_name = Path(video_path).stem
            output_json_path = os.path.join(video_dir, f"{video_name}_transcription.json")
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("PROCESSING RESULTS")
        print(f"{'='*60}")
        print(f"Video File: {video_path}")
        print(f"Audio File (WAV): {temp_audio_path}")
        print(f"Detected Language: {language_code.upper()} (confidence: {confidence:.2%})")
        print(f"Classification: {'Indian' if is_indian else 'Non-Indian'}")
        print(f"Transcription Model: {transcription_result.get('model', 'unknown')}")
        print(f"Transcription Text: {transcription_result.get('text', '')[:100]}...")
        print(f"Output JSON: {output_json_path}")
        print(f"{'='*60}\n")
        
        return output_data
        
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
        print("Usage: python video_lid_ecapa_indic_seamless_pipeline.py <video_file_path> [output_json_path] [output_wav_path]")
        print("\nExample:")
        print("  python video_lid_ecapa_indic_seamless_pipeline.py /path/to/video.mp4")
        print("  python video_lid_ecapa_indic_seamless_pipeline.py /path/to/video.mp4 /path/to/output.json")
        print("  python video_lid_ecapa_indic_seamless_pipeline.py /path/to/video.mp4 /path/to/output.json /path/to/output.wav")
        print("\nPipeline:")
        print("  1. Extract audio from video")
        print("  2. Detect language using ECAPA-TDNN")
        print("  3. Classify as Indian or Non-Indian")
        print("  4. Transcribe with Indic Seamless (Indian) or Whisper (Non-Indian)")
        print("  5. Save results to JSON")
        print("\nIndic Seamless: https://model.aibase.com/models/details/1915693340919750658")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_wav_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Convert to absolute path
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(video_path)
    
    try:
        result = process_video_pipeline(
            video_path,
            output_json_path=output_json_path,
            keep_audio=(output_wav_path is not None),
            output_wav_path=output_wav_path
        )
        print(f"\n✓ Processing completed successfully!")
        print(f"  Detected Language: {result['language_detection']['language_code'].upper()}")
        print(f"  Classification: {'Indian' if result['language_detection']['is_indian'] else 'Non-Indian'}")
        print(f"  Transcription Model: {result['transcription']['model']}")
        print(f"  Output JSON: {output_json_path or 'default location'}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

