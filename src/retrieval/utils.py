"""
Utility functions for retrieval operations
"""
from typing import List, Any


def merge_overlapping_clips(search_results: List[Any], gap_seconds: int = 3) -> List[Any]:
    """
    Merges overlapping or nearby video clips from the same video.
    
    Args:
        search_results: List of search result objects with metadata containing:
                       - video_doc_id: ID of the video
                       - start_time: Start time in seconds
                       - end_time: End time in seconds
                       - score: Similarity score
                       - source: Source modalities (optional)
        gap_seconds: Maximum gap in seconds between clips to consider them mergeable
        
    Returns:
        List of merged search results, sorted by score (descending)
    """
    if not search_results:
        return []
    
    # Group results by video
    results_by_video = {}
    for res in search_results:
        # Handle both dict-like and object-like results
        if hasattr(res, 'metadata'):
            metadata = res.metadata
            score = res.score
        else:
            metadata = res.get('metadata', res)
            score = res.get('score', 0.0)
        
        video_id = metadata.get("video_doc_id") if isinstance(metadata, dict) else getattr(metadata, 'video_doc_id', None)
        if video_id not in results_by_video:
            results_by_video[video_id] = []
        results_by_video[video_id].append(res)
    
    final_merged_results = []
    for video_id, clips in results_by_video.items():
        if not clips:
            continue
        
        # Sort clips by start time to easily check for overlaps
        def get_start_time(clip):
            if hasattr(clip, 'metadata'):
                return clip.metadata.get('start_time', 0)
            return clip.get('metadata', {}).get('start_time', clip.get('start_time', 0))
        
        clips.sort(key=get_start_time)
        
        merged_for_video = []
        if not clips:
            continue
        current_merge = clips[0]
        
        for next_clip in clips[1:]:
            # Get metadata and scores
            def get_metadata(clip):
                if hasattr(clip, 'metadata'):
                    return clip.metadata
                return clip.get('metadata', clip)
            
            def get_score(clip):
                if hasattr(clip, 'score'):
                    return clip.score
                return clip.get('score', 0.0)
            
            current_meta = get_metadata(current_merge)
            next_meta = get_metadata(next_clip)
            
            current_end = current_meta.get('end_time', 0) if isinstance(current_meta, dict) else getattr(current_meta, 'end_time', 0)
            next_start = next_meta.get('start_time', 0) if isinstance(next_meta, dict) else getattr(next_meta, 'start_time', 0)
            
            # Check for overlap or if the next clip is within the gap
            if next_start <= current_end + gap_seconds:
                # Merge clips - extend end time
                if isinstance(current_meta, dict):
                    current_meta['end_time'] = max(current_end, next_meta.get('end_time', 0) if isinstance(next_meta, dict) else getattr(next_meta, 'end_time', 0))
                else:
                    current_meta.end_time = max(current_end, next_meta.get('end_time', 0) if isinstance(next_meta, dict) else getattr(next_meta, 'end_time', 0))
                
                # Keep the highest score
                current_score = get_score(current_merge)
                next_score = get_score(next_clip)
                if next_score > current_score:
                    if hasattr(current_merge, 'score'):
                        current_merge.score = next_score
                    else:
                        current_merge['score'] = next_score
                
                # Combine source information
                current_source_str = current_meta.get('source', '') if isinstance(current_meta, dict) else getattr(current_meta, 'source', '')
                next_source_str = next_meta.get('source', '') if isinstance(next_meta, dict) else getattr(next_meta, 'source', '')
                
                current_source = set(str(current_source_str).split(', '))
                next_source = set(str(next_source_str).split(', '))
                current_source.discard('')
                next_source.discard('')
                
                combined_sources = sorted(list(current_source.union(next_source)))
                source_str = ', '.join(combined_sources)
                
                if isinstance(current_meta, dict):
                    current_meta['source'] = source_str
                else:
                    current_meta.source = source_str
            else:
                # No overlap, finalize the current merged clip and start a new one
                merged_for_video.append(current_merge)
                current_merge = next_clip
        
        merged_for_video.append(current_merge)  # Add the last merged clip
        final_merged_results.extend(merged_for_video)
    
    # Sort the final results by score again
    def get_score_for_sort(clip):
        if hasattr(clip, 'score'):
            return clip.score
        return clip.get('score', 0.0)
    
    final_merged_results.sort(key=get_score_for_sort, reverse=True)
    return final_merged_results

