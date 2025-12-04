"""
Retrieval functionality for video search and analysis
"""
from .search import (
    search_videos,
    perform_text_search,
    perform_image_search,
    perform_audio_search,
    query_index
)
from .analyze import (
    analyze_video,
    get_query_intent,
    get_llm_analysis_json,
    format_transcript_context
)
from .scoring import (
    compute_final_score,
    compute_audio_search_score
)
from .utils import (
    merge_overlapping_clips
)

__all__ = [
    # Search functions
    'search_videos',
    'perform_text_search',
    'perform_image_search',
    'perform_audio_search',
    'query_index',
    # Analysis functions
    'analyze_video',
    'get_query_intent',
    'get_llm_analysis_json',
    'format_transcript_context',
    # Scoring functions
    'compute_final_score',
    'compute_audio_search_score',
    # Utility functions
    'merge_overlapping_clips',
]
