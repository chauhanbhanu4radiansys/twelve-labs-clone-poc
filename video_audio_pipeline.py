"""
Video to Audio Pipeline with IndicLID and Whisper
Processes video files: extracts audio, identifies language, transcribes and translates.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import whisper
except ImportError:
    raise ImportError("Please install openai-whisper: pip install openai-whisper")

try:
    from transformers import AutoModelForAudioClassification, AutoProcessor
    import torch
    import torchaudio
except ImportError:
    raise ImportError("Please install transformers, torch, and torchaudio: pip install transformers torch torchaudio")


# IndicLID model configuration
INDICLID_MODEL_NAME = "ai4bharat/IndicLID"
INDIAN_LANGUAGE_CODES = {
    "hi": "hindi", "bn": "bengali", "pa": "punjabi", "gu": "gujarati",
    "mr": "marathi", "ml": "malayalam", "ta": "tamil", "te": "telugu",
    "kn": "kannada", "or": "oriya", "as": "assamese", "ks": "kashmiri",
    "mai": "maithili", "sd": "sindhi", "ne": "nepali", "si": "sinhala",
    "en": "english"
}


def extract_audio_from_video(video_path: str, output_audio_path: Optional[str] = None) -> str:
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
    # -ar 16000: sample rate 16kHz (required for IndicLID)
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
        print(f"Audio extracted successfully to: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html")


def load_indiclid_model():
    """
    Loads the IndicLID model from Hugging Face.
    Note: IndicLID may be a NeMo model. If this fails, you may need to use NeMo's EncDecClassificationModel.
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading IndicLID model: {INDICLID_MODEL_NAME}")
    
    # Try loading as a standard Hugging Face transformers model first
    try:
        processor = AutoProcessor.from_pretrained(INDICLID_MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(INDICLID_MODEL_NAME)
        model.eval()
        print("IndicLID model loaded successfully from Hugging Face")
        return model, processor
    except Exception as e1:
        print(f"Failed to load as standard Hugging Face model: {e1}")
        print("Attempting to load as NeMo model...")
        
        # Fallback: Try loading as NeMo model
        try:
            from nemo.collections.asr.models import EncDecClassificationModel
            model = EncDecClassificationModel.from_pretrained(model_name=INDICLID_MODEL_NAME)
            model.eval()
            print("IndicLID model loaded successfully from NeMo")
            # Return None for processor since NeMo models don't use AutoProcessor
            return model, None
        except ImportError:
            raise RuntimeError(
                f"Failed to load IndicLID model. "
                f"IndicLID appears to be a NeMo model, not a standard Hugging Face transformers model. "
                f"Please install NeMo: pip install nemo_toolkit[asr]"
            )
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load IndicLID model from both Hugging Face and NeMo. "
                f"Hugging Face error: {e1}. NeMo error: {e2}"
            )


def identify_language_with_indiclid(audio_path: str, model, processor) -> Tuple[str, bool]:
    """
    Identifies language using IndicLID model.
    Supports both Hugging Face transformers models and NeMo models.
    
    Args:
        audio_path: Path to the audio WAV file
        model: Loaded IndicLID model (Hugging Face or NeMo)
        processor: IndicLID processor (None for NeMo models)
    
    Returns:
        Tuple of (language_code, is_indian)
    """
    print(f"Identifying language for audio: {audio_path}")
    
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed (IndicLID requirement)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Handle NeMo model (processor is None)
        if processor is None:
            # NeMo model API
            # The model expects a batch dimension
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            model.eval()
            with torch.no_grad():
                logits = model(input_signal=waveform, input_signal_length=torch.tensor([waveform.shape[1]]))
                
                # The output contains logits, we need to find the predicted label index
                if isinstance(logits, tuple):
                    logits = logits[0]
                    
                predicted_index = torch.argmax(logits, dim=1)
                language = model.cfg.labels[predicted_index.item()]
            
            language_code = language.lower().strip()
        
        else:
            # Hugging Face transformers model API
            audio_array = waveform.squeeze().numpy()
            
            # Use processor to prepare inputs
            if hasattr(processor, '__call__'):
                # Try direct call first
                try:
                    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
                except Exception:
                    # Fallback: manual processing
                    inputs = {"input_values": torch.from_numpy(audio_array).unsqueeze(0)}
            else:
                inputs = {"input_values": torch.from_numpy(audio_array).unsqueeze(0)}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Get predicted label
            predicted_id = torch.argmax(logits, dim=-1).item()
            
            # Get label name from model config
            if hasattr(model.config, 'id2label') and model.config.id2label:
                label = model.config.id2label[predicted_id]
            elif hasattr(model.config, 'label2id') and model.config.label2id:
                # Reverse lookup
                label2id = model.config.label2id
                label = [k for k, v in label2id.items() if v == predicted_id][0] if label2id else str(predicted_id)
            else:
                # Fallback: try to get from labels attribute or use index
                label = str(predicted_id)
            
            # Extract language code (assuming format like "hi", "en", etc.)
            # Handle different label formats
            language_code = str(label).lower().strip()
            
            # Remove common prefixes/suffixes if present
            if language_code.startswith('lang_'):
                language_code = language_code[5:]
            if '_' in language_code:
                language_code = language_code.split('_')[0]
        
        # Check if it's an Indian language (excluding English)
        is_indian = language_code in INDIAN_LANGUAGE_CODES and language_code != "en"
        
        print(f"Detected language: {language_code}, Is Indian: {is_indian}")
        return language_code, is_indian
        
    except Exception as e:
        print(f"Error during language identification: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: assume non-Indian
        return "en", False


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


def process_video_pipeline(video_path: str, output_json_path: Optional[str] = None) -> Dict:
    """
    Main pipeline: Video -> Audio -> IndicLID -> Transcription -> Translation
    
    Args:
        video_path: Full path to the input video file
        output_json_path: Optional path for output JSON file. If None, creates file next to video.
    
    Returns:
        Dictionary containing all processing results
    """
    print(f"Starting pipeline for video: {video_path}")
    
    # Step 1: Extract audio from video
    temp_audio_path = None
    try:
        temp_audio_path = extract_audio_from_video(video_path)
        
        # Step 2: Load IndicLID model
        indiclid_model, indiclid_processor = load_indiclid_model()
        
        # Step 3: Identify language
        language_code, is_indian = identify_language_with_indiclid(
            temp_audio_path,
            indiclid_model,
            indiclid_processor
        )
        
        # Step 4: Transcribe with Whisper
        # Load Whisper model once and reuse for transcription and translation
        whisper_model = whisper.load_model("base")
        
        if is_indian:
            print(f"Processing Indian language: {language_code}")
            # Use detected language code for transcription
            transcription_result = transcribe_with_whisper(temp_audio_path, language=language_code, model=whisper_model)
        else:
            print("Processing non-Indian language")
            # Auto-detect language
            transcription_result = transcribe_with_whisper(temp_audio_path, language=None, model=whisper_model)
        
        # Step 5: Translate segments to English
        segments = transcription_result.get('segments', [])
        translated_segments = translate_segments_with_whisper(
            temp_audio_path,
            segments,
            language=language_code if is_indian else None,
            model=whisper_model
        )
        
        # Prepare output
        output_data = {
            "video_path": video_path,
            "audio_path": temp_audio_path,
            "language_detection": {
                "language_code": language_code,
                "is_indian": is_indian,
                "language_name": INDIAN_LANGUAGE_CODES.get(language_code, language_code)
            },
            "transcription": {
                "text": transcription_result.get('text', ''),
                "language": transcription_result.get('language', language_code),
                "segments": translated_segments
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
        
        print(f"Results saved to: {output_json_path}")
        
        return output_data
        
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
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_audio_pipeline.py <video_file_path> [output_json_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = process_video_pipeline(video_path, output_json_path)
        print("\nPipeline completed successfully!")
        print(f"Language detected: {result['language_detection']['language_name']}")
        print(f"Total segments: {len(result['transcription']['segments'])}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

