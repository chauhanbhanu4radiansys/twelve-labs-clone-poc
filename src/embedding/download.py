"""
Download utilities for pre-signed S3 URLs
"""
import os
import tempfile
import requests
from typing import Optional


def download_from_url(url: str, output_path: Optional[str] = None, chunk_size: int = 8192) -> Optional[str]:
    """
    Downloads a file from a pre-signed S3 URL.
    
    Args:
        url: Pre-signed S3 URL
        output_path: Optional output path. If None, creates a temporary file.
        chunk_size: Chunk size for streaming download
        
    Returns:
        Path to downloaded file, or None if download failed
    """
    try:
        if output_path is None:
            # Create temporary file
            file_ext = os.path.splitext(url.split('?')[0])[1] or '.tmp'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            output_path = temp_file.name
            temp_file.close()
        
        # Download with streaming
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        return output_path
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None


def download_transcript_from_url(url: str) -> Optional[list]:
    """
    Downloads and parses a transcript JSON from a pre-signed S3 URL.
    
    Args:
        url: Pre-signed S3 URL to transcript JSON file
        
    Returns:
        List of transcript segments, or None if download/parse failed
    """
    import json
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        transcript_data = response.json()
        
        # Handle different transcript formats
        if isinstance(transcript_data, list):
            return transcript_data
        elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
            return transcript_data['segments']
        elif isinstance(transcript_data, dict) and 'transcript' in transcript_data:
            return transcript_data['transcript']
        else:
            print(f"Warning: Unexpected transcript format. Returning as-is.")
            return transcript_data if isinstance(transcript_data, list) else None
            
    except Exception as e:
        print(f"Error downloading/parsing transcript from URL: {e}")
        return None

