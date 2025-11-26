"""
Embedding generation pipeline for video processing
"""

from .pipeline import process_video
from .download import download_from_url, download_transcript_from_url

__all__ = ['process_video', 'download_from_url', 'download_transcript_from_url']

