"""
Video Language Identification Pipeline using Vakgyata Base Model
Extracts audio from video using ffmpeg (WAV format), identifies language
using onecxi/vakgyata-base model, transcribes with Whisper, and translates to English.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
except ImportError:
    raise ImportError(
        "Please install required packages:\n"
        "pip install torch torchaudio transformers"
    )

try:
    import whisper
except ImportError:
    raise ImportError(
        "Please install openai-whisper:\n"
        "pip install openai-whisper"
    )

# Language code mapping from Vakgyata to Whisper language codes
# Vakgyata uses language labels that may need mapping to Whisper's ISO codes
LANGUAGE_CODE_MAP = {
    "hindi": "hi",
    "english": "en",
    "bengali": "bn",
    "telugu": "te",
    "marathi": "mr",
    "tamil": "ta",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "oriya": "or",
    "assamese": "as",
    "urdu": "ur",
    "nepali": "ne",
    "sinhala": "si",
    "sindhi": "sd",
    "kashmiri": "ks",
    "maithili": "mai",
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


def map_language_to_whisper(language_code: str) -> Optional[str]:
    """
    Maps Vakgyata language code to Whisper language code.
    
    Args:
        language_code: Language code from Vakgyata model
    
    Returns:
        Whisper language code or None if not found
    """
    # Normalize language code (lowercase, strip)
    lang_lower = language_code.lower().strip()
    
    # Check direct mapping
    if lang_lower in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[lang_lower]
    
    # If already a 2-letter code, return as is (Whisper uses ISO 639-1)
    if len(lang_lower) == 2:
        return lang_lower
    
    # Return None to let Whisper auto-detect
    return None


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
            result = model.transcribe(audio_path, language=language)
        else:
            result = model.transcribe(audio_path)
        
        print(f"Transcription completed. Found {len(result.get('segments', []))} segments")
        return result
        
    except Exception as e:
        raise RuntimeError(f"Whisper transcription error: {e}")


def translate_segments_with_whisper(audio_path: str, segments: List[Dict], language: Optional[str] = None, model: Optional[Any] = None) -> List[Dict]:
    """
    Translates each segment to English using Whisper.
    
    Args:
        audio_path: Path to the audio file
        segments: List of segment dictionaries with 'start', 'end', and 'text'
        language: Source language code
        model: Optional pre-loaded Whisper model (for efficiency)
    
    Returns:
        List of segments with added 'translation' field
    """
    print(f"Translating {len(segments)} segments to English")
    
    try:
        # Load Whisper model if not provided
        if model is None:
            model = whisper.load_model("base")
        
        translated_segments = []
        
        # Translate each segment individually to get accurate segment-level translations
        for idx, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            
            print(f"Translating segment {idx + 1}/{len(segments)} ({start_time:.2f}s - {end_time:.2f}s)")
            
            # Extract segment audio using ffmpeg
            temp_segment_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_segment_path.close()
            
            try:
                # Extract segment
                cmd = [
                    "ffmpeg",
                    "-i", audio_path,
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    "-loglevel", "error",  # Suppress ffmpeg output
                    temp_segment_path.name
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                
                # Verify segment file exists and has content
                if not os.path.exists(temp_segment_path.name) or os.path.getsize(temp_segment_path.name) < 1000:
                    print(f"Warning: Segment {idx + 1} audio file is too small or missing, skipping translation")
                    segment_copy = segment.copy()
                    segment_copy['translation'] = ""
                    translated_segments.append(segment_copy)
                    continue
                
                # Translate segment
                if language and language != "en":
                    translation_result = model.transcribe(
                        temp_segment_path.name,
                        language=language,
                        task="translate"
                    )
                else:
                    translation_result = model.transcribe(
                        temp_segment_path.name,
                        task="translate"
                    )
                
                translated_text = translation_result.get('text', '').strip()
                
                # Add translation to segment
                segment_copy = segment.copy()
                segment_copy['translation'] = translated_text
                translated_segments.append(segment_copy)
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to extract segment {idx + 1}: {e.stderr.decode() if e.stderr else str(e)}")
                segment_copy = segment.copy()
                segment_copy['translation'] = ""
                translated_segments.append(segment_copy)
            except Exception as e:
                print(f"Warning: Error translating segment {idx + 1}: {e}")
                segment_copy = segment.copy()
                segment_copy['translation'] = ""
                translated_segments.append(segment_copy)
            finally:
                # Clean up temp file
                if os.path.exists(temp_segment_path.name):
                    try:
                        os.unlink(temp_segment_path.name)
                    except Exception:
                        pass
        
        print("Translation completed")
        return translated_segments
        
    except Exception as e:
        raise RuntimeError(f"Whisper translation error: {e}")


def process_video_pipeline(video_path: str, output_json_path: Optional[str] = None, keep_audio: bool = False, output_wav_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete pipeline: Extract audio -> LID -> Transcription -> Translation -> Save to JSON.
    
    Args:
        video_path: Path to the input video file
        output_json_path: Optional path for output JSON file. If None, creates file next to video.
        keep_audio: If True, keeps the extracted WAV file. If False, deletes it after processing.
        output_wav_path: Optional path for output WAV file. If None, creates temp file.
    
    Returns:
        Dictionary containing all processing results
    """
    print(f"\n{'='*60}")
    print(f"Processing video: LID -> Transcription -> Translation")
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
        
        # Step 4: Map language code to Whisper format
        whisper_lang = map_language_to_whisper(language_code)
        print(f"Mapped language '{language_code}' to Whisper code: {whisper_lang or 'auto-detect'}")
        
        # Step 5: Load Whisper model once and reuse for transcription and translation
        whisper_model = whisper.load_model("base")
        
        # Step 6: Transcribe with Whisper
        transcription_result = transcribe_with_whisper(
            temp_audio_path,
            language=whisper_lang,
            model=whisper_model
        )
        
        # Step 7: Translate segments to English
        segments = transcription_result.get('segments', [])
        translated_segments = translate_segments_with_whisper(
            temp_audio_path,
            segments,
            language=whisper_lang,
            model=whisper_model
        )
        
        # Prepare full English translation text (concatenate all translations)
        full_translation_text = " ".join([
            seg.get('translation', '') 
            for seg in translated_segments 
            if seg.get('translation', '').strip()
        ]).strip()
        
        # Prepare output
        output_data = {
            "video_path": video_path,
            "audio_path": temp_audio_path,
            "language_detection": {
                "language_code": language_code,
                "whisper_language_code": whisper_lang,
                "confidence": confidence,
                "language_index": language_index
            },
            "transcription": {
                "text": transcription_result.get('text', ''),
                "language": transcription_result.get('language', whisper_lang or 'auto'),
                "segments": [
                    {
                        "start": seg.get('start', 0.0),
                        "end": seg.get('end', 0.0),
                        "text": seg.get('text', ''),
                        "translation": seg.get('translation', '')
                    }
                    for seg in translated_segments
                ]
            },
            "translation": {
                "full_text": full_translation_text,
                "language": "en",
                "segments": [
                    {
                        "start": seg.get('start', 0.0),
                        "end": seg.get('end', 0.0),
                        "text": seg.get('translation', '')
                    }
                    for seg in translated_segments
                    if seg.get('translation', '').strip()
                ]
            }
        }
        
        # Step 8: Write to JSON file
        if output_json_path is None:
            video_dir = os.path.dirname(video_path)
            video_name = Path(video_path).stem
            output_json_path = os.path.join(video_dir, f"{video_name}_transcription.json")
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        # Save main JSON file with all data
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Step 9: Save separate translations file (JSON format)
        translations_json_path = output_json_path.replace('.json', '_translations.json')
        translations_data = {
            "video_path": video_path,
            "source_language": language_code,
            "target_language": "en",
            "full_translation": full_translation_text,
            "segments": output_data["translation"]["segments"]
        }
        with open(translations_json_path, 'w', encoding='utf-8') as f:
            json.dump(translations_data, f, indent=2, ensure_ascii=False)
        
        # Step 10: Save translations as plain text file
        translations_txt_path = output_json_path.replace('.json', '_translations.txt')
        with open(translations_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"English Translation\n")
            f.write(f"{'='*60}\n")
            f.write(f"Source Language: {language_code.upper()}\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"{'='*60}\n\n")
            f.write(full_translation_text)
            f.write("\n")
        
        print(f"\n{'='*60}")
        print("PROCESSING RESULTS")
        print(f"{'='*60}")
        print(f"Video File: {video_path}")
        print(f"Audio File (WAV): {temp_audio_path}")
        print(f"Detected Language: {language_code} (confidence: {confidence:.2%})")
        print(f"Transcription Language: {transcription_result.get('language', 'auto')}")
        print(f"Total Segments: {len(translated_segments)}")
        print(f"Output JSON: {output_json_path}")
        print(f"Translations JSON: {translations_json_path}")
        print(f"Translations TXT: {translations_txt_path}")
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
        print("Usage: python video_lid_vakgyata_pipeline.py <video_file_path> [output_json_path] [output_wav_path]")
        print("\nExample:")
        print("  python video_lid_vakgyata_pipeline.py /path/to/video.mp4")
        print("  python video_lid_vakgyata_pipeline.py /path/to/video.mp4 /path/to/output.json")
        print("  python video_lid_vakgyata_pipeline.py /path/to/video.mp4 /path/to/output.json /path/to/output.wav")
        print("\nModel: onecxi/vakgyata-base (designed for Indian languages)")
        print("Pipeline: LID -> Whisper Transcription -> English Translation -> JSON Output")
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
        print(f"  Detected Language: {result['language_detection']['language_code']}")
        print(f"  Confidence: {result['language_detection']['confidence']:.2%}")
        print(f"  Transcription Segments: {len(result['transcription']['segments'])}")
        print(f"  Output JSON: {output_json_path or 'default location'}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

