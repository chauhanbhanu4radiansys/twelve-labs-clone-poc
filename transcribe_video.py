#!/usr/bin/env python3
"""
Video transcription script using OpenAI Whisper.

This script takes an absolute path to a video file and uses OpenAI Whisper
to create a transcript, saving it in the same directory as the video.
"""

import os
import sys
import argparse
from pathlib import Path


def transcribe_video(video_path: str, model_size: str = "base", output_format: str = "json") -> str:
    """
    Transcribe a video file using OpenAI Whisper.
    
    Args:
        video_path: Absolute path to the video file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_format: Output format (txt, srt, vtt, json, tsv)
    
    Returns:
        Path to the saved transcript file
    """
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper is not installed.", file=sys.stderr)
        print("Please install it with: pip install openai-whisper", file=sys.stderr)
        sys.exit(1)
    
    # Validate video path
    if not os.path.isabs(video_path):
        print(f"Error: Path must be absolute. Got: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    # Get video directory and base name
    video_dir = os.path.dirname(video_path)
    video_name = Path(video_path).stem
    
    # Load Whisper model
    print(f"Loading Whisper model '{model_size}'...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"Error loading Whisper model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Transcribe video
    print(f"Transcribing video: {os.path.basename(video_path)}")
    print("This may take a while depending on video length...")
    
    try:
        result = model.transcribe(video_path)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file path
    output_extensions = {
        "txt": ".txt",
        "srt": ".srt",
        "vtt": ".vtt",
        "json": ".json",
        "tsv": ".tsv"
    }
    
    if output_format not in output_extensions:
        print(f"Warning: Unknown output format '{output_format}'. Using 'json'.", file=sys.stderr)
        output_format = "json"
    
    output_ext = output_extensions[output_format]
    output_path = os.path.join(video_dir, f"{video_name}_transcript{output_ext}")
    
    # Save transcript based on format
    try:
        if output_format == "txt":
            # Save plain text transcript
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
        elif output_format == "srt":
            # Save as SRT subtitle format
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], start=1):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
        elif output_format == "vtt":
            # Save as WebVTT format
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start_time = format_timestamp_vtt(segment["start"])
                    end_time = format_timestamp_vtt(segment["end"])
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
        elif output_format == "json":
            # Save full JSON result
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif output_format == "tsv":
            # Save as TSV (tab-separated values)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("start\tend\ttext\n")
                for segment in result["segments"]:
                    f.write(f"{segment['start']}\t{segment['end']}\t{segment['text'].strip()}\n")
        
        print(f"\nTranscript saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error saving transcript: {e}", file=sys.stderr)
        sys.exit(1)


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Transcribe a video file using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_video.py /path/to/video.mp4
  python transcribe_video.py /path/to/video.mp4 --model large --format srt
        """
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Absolute path to the video file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger models are more accurate but slower."
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["txt", "srt", "vtt", "json", "tsv"],
        help="Output format (default: json). Options: txt (plain text), srt (subtitles), vtt (WebVTT), json (full result with segments), tsv (tab-separated)"
    )
    
    args = parser.parse_args()
    
    transcribe_video(args.video_path, args.model, args.format)


if __name__ == "__main__":
    main()
