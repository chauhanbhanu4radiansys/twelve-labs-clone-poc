#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
import transformers
print(transformers.__version__)
print([cls for cls in dir(transformers) if "SeamlessM4T" in cls])


# --- Model Config ---
MODEL_ID = "facebook/seamless-m4t-v2-large"
TARGET_SR = 16000

SUPPORTED_LANGUAGES = {
    "as","bn","gu","hi","mr","or","pa","ur",
    "kn","kok","ks","mai","ml","mni","ne",
    "sa","sat","sd","ta","te","brx","doi"
}

def extract_audio_ffmpeg(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_wav = tmp.name

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-y",
        "-loglevel", "error",
        output_wav
    ]
    subprocess.run(cmd, check=True)
    return output_wav

def load_model():
    print(f"Loading model: {MODEL_ID} …")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

    print("✓ Model & processor loaded.")
    return processor, model

def transcribe_audio(wav_path: str, processor, model, lang_code: str) -> str:
    waveform, sr = torchaudio.load(wav_path)
    print(f"Loaded waveform: shape={waveform.shape}, sample_rate={sr}")

    # convert to mono if needed
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # resample if needed
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(
            waveform
        )

    waveform = waveform.squeeze(0).numpy()  # [T]

    # Prepare inputs for the model
    inputs = processor(audios=waveform, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(model.device)

    # Generate transcription (ASR) — specify target language code
    # For ASR, we use tgt_lang = language code
    generated_ids = model.generate(
        input_features=input_features,
        tgt_lang=lang_code,
        generate_speech=False   # ensures we ask for text output not speech
    )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <video_path.mp4> <language_code>")
        print("Supported languages:", ", ".join(sorted(SUPPORTED_LANGUAGES)))
        sys.exit(1)

    video_path = sys.argv[1]
    lang_code = sys.argv[2].lower()

    if lang_code not in SUPPORTED_LANGUAGES:
        print(f"Error: language '{lang_code}' not supported.")
        sys.exit(1)

    print("--- Starting transcription pipeline ---")
    print(f"Video: {video_path}")
    print(f"Language code: {lang_code}")

    wav_path = extract_audio_ffmpeg(video_path)
    try:
        processor, model = load_model()
        transcription = transcribe_audio(wav_path, processor, model, lang_code)

        output_txt = Path(video_path).with_suffix(".txt")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(transcription)

        print("\n--- TRANSCRIPTION ---")
        print(transcription)
        print("---------------------")
        print(f"✓ Saved to: {output_txt}")
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                print(f"Cleaned up temp file: {wav_path}")
            except Exception as e:
                print(f"Warning: could not delete {wav_path}: {e}")

if __name__ == "__main__":
    main()
