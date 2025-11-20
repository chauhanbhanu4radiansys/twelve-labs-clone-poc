import torch
import librosa
import warnings
from speechbrain.pretrained import EncoderClassifier
from transformers import AutoProcessor, AutoModelForCTC, pipeline

# --- Global Settings ---

# Suppress harmless warnings
warnings.filterwarnings("ignore")

# Set device (will automatically use Colab's GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Define the sample rate all models expect
TARGET_SAMPLE_RATE = 16000

# Map of detected lang codes to their specific AI4Bharat Conformer model
INDIAN_LANG_MODEL_MAP = {
    "hi": "ai4bharat/indic-conformer-asr-hi",  # Hindi
    "ta": "ai4bharat/indic-conformer-asr-ta",  # Tamil
    "te": "ai4bharat/indic-conformer-asr-te",  # Telugu
    "mr": "ai4bharat/indic-conformer-asr-mr",  # Marathi
    "bn": "ai4bharat/indic-conformer-asr-bn",  # Bengali
    "gu": "ai4bharat/indic-conformer-asr-gu",  # Gujarati
    "kn": "ai4bharat/indic-conformer-asr-kn",  # Kannada
    "ml": "ai4bharat/indic-conformer-asr-ml",  # Malayalam
    "or": "ai4bharat/indic-conformer-asr-or",  # Odia
    "pa": "ai4bharat/indic-conformer-asr-pa",  # Punjabi
    "as": "ai4bharat/indic-conformer-asr-as",  # Assamese
}

# Set of Indian language codes for quick lookup
INDIAN_LANG_CODES = set(INDIAN_LANG_MODEL_MAP.keys())

# --- Model Functions ---

def load_audio(file_path):
    """
    Loads and resamples a WAV file to the target sample rate.
    This is used for the transcription models (Whisper/Conformer).
    """
    try:
        # Load audio file using librosa
        audio_input, original_sample_rate = librosa.load(file_path, sr=None)

        # Resample if necessary
        if original_sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {original_sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz...")
            audio_input = librosa.resample(
                audio_input, 
                orig_sr=original_sample_rate, 
                target_sr=TARGET_SAMPLE_RATE
            )
        
        print(f"Audio file '{file_path}' loaded successfully for transcription.")
        return audio_input, TARGET_SAMPLE_RATE
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def detect_language(audio_file_path):
    """
    Detects the language using speechbrain/lang-id-voxlingua107-ecapa.
    This function now takes a file path and uses the model's internal loader.
    """
    print("Detecting language with ECAPA-TDNN (voxlingua107)...")
    try:
        # Load the language ID model and move it to the active device
        lang_id_model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa", 
            savedir="tmp_lang_id_cache"
        ).to(DEVICE)

        # --- THIS IS THE FIX ---
        # Use the model's own .load_audio() method.
        # It handles loading and resampling, but we need to explicitly
        # move the tensor to the same device as the model.
        signal = lang_id_model.load_audio(audio_file_path)
        
        # Explicitly move signal to the correct device (CPU/GPU mismatch fix)
        signal = signal.to(DEVICE)
        
        # Perform classification.
        # 'signal' is now on the correct device and has batch dim [1, n_samples]
        prediction = lang_id_model.classify_batch(signal)
        # ---------------------
        
        # Extract the top predicted language label
        lang_label = prediction[3][0]
        # Extract the 2-letter code (e.g., 'hi')
        lang_code = lang_label.split(':')[0].strip()
        
        print(f"Detected language: {lang_label} (Code: {lang_code})")
        return lang_code
    except Exception as e:
        print(f"Error during language detection: {e}")
        return None

def transcribe_indian(audio_array, model_id):
    """Transcribes using a specified AI4Bharat Conformer model."""
    print(f"Transcribing with Indian model: {model_id} ...")
    try:
        # Load the processor and model for CTC
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id).to(DEVICE)

        # Process the audio array
        inputs = processor(
            audio_array, 
            sampling_rate=TARGET_SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        ).to(DEVICE)

        # Run inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        # Decode the model output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
    except Exception as e:
        print(f"Error during Indian model transcription: {e}")
        return "[Transcription failed]"

def transcribe_whisper(audio_array, sample_rate):
    """Transcribes using OpenAI Whisper."""
    print("Transcribing with OpenAI Whisper (large-v3 model)...")
    try:
        # Use the pipeline for a simple ASR setup
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=DEVICE
        )

        # Pass the raw audio array and sample rate
        input_data = {"raw": audio_array, "sampling_rate": sample_rate}
        
        # Set chunk_length_s for long audio processing
        result = whisper_pipe(input_data, chunk_length_s=30, batch_size=8)
        
        return result["text"]
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return "[Transcription failed]"

# --- Main Execution Function ---

def main(audio_file):
    # 1. Detect language FIRST, using the file path
    # This uses the model's internal .load_audio() to avoid device mismatch
    lang_code = detect_language(audio_file)
    if lang_code is None:
        print("Could not detect language. Exiting.")
        return

    # 2. Load and prepare audio array *after* detection
    # This is for the transcription models (Conformer/Whisper)
    audio_input, sample_rate = load_audio(audio_file)
    if audio_input is None:
        print("Could not load audio for transcription. Exiting.")
        return

    # 3. Conditional transcription
    transcription = ""
    if lang_code in INDIAN_LANG_CODES:
        # Use the specific Indian language model
        model_id = INDIAN_LANG_MODEL_MAP[lang_code]
        transcription = transcribe_indian(audio_input, model_id)
    else:
        # Use OpenAI Whisper for all other languages
        transcription = transcribe_whisper(audio_input, sample_rate)

    # 4. Print the final result
    print("\n" + "="*30)
    print("      FINAL TRANSCRIPTION")
    print("="*30)
    print(transcription)
    print("="*30)


# --- 🚀 RUN THE SCRIPT HERE ---

# <-- !! IMPORTANT !! SET YOUR FILE PATH HERE
# 1. Upload your .wav file to Colab (drag-and-drop to the left panel).
# 2. Change the path below to match your file's name.
#    If you just upload it, the path will be "/content/your_file_name.wav"

INPUT_AUDIO_FILE = "pvid1.wav" 

# -------------------------------------

try:
    main(INPUT_AUDIO_FILE)
except FileNotFoundError:
    print("\n" + "="*50)
    print(f"❌ ERROR: File not found at '{INPUT_AUDIO_FILE}'")
    print("Please make sure you have uploaded the file and the path is correct.")
    print("="*50)
except Exception as e:
    print(f"An unexpected error occurred: {e}")