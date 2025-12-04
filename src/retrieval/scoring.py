"""
Scoring functions for multi-modal search results
"""
from typing import Dict


def compute_final_score(modality_scores: Dict[str, float], threshold: float = 0.75) -> float:
    """
    Computes a final weighted score from multiple modality scores.
    
    Args:
        modality_scores: Dictionary with modality names as keys and scores as values
                         e.g., {'video': 0.33, 'audio': 0.36, 'text': 0.28, 'desc': 0.40}
        threshold: If any raw score >= threshold, use that score directly
        
    Returns:
        Final weighted score (float, rounded to 3 decimal places)
    """
    # Check for any raw score ≥ threshold
    if not modality_scores:
        return 0.0
    
    max_raw_score = max(modality_scores.values())
    if max_raw_score >= threshold:
        return round(max_raw_score, 3)  # Use raw score directly
    
    # Compute base fused score (weighted average)
    weights = {"video": 1.0, "desc": 0.9, "audio": 0.8, "text": 0.6}
    
    # Filter out modalities not present in the scores
    present_modalities = [m for m in weights if m in modality_scores]
    if not present_modalities:
        return 0.0
    
    total_w = sum(weights[m] for m in present_modalities)
    base_score = sum(modality_scores[m] * weights[m] for m in present_modalities) / total_w
    
    # Identify top-priority modality present
    priority_order = ["video", "desc", "audio", "text"]
    top_mod = next((m for m in priority_order if m in present_modalities), None)
    top_weight = weights[top_mod] if top_mod else 0.6  # Fallback if something odd
    
    # Apply priority boost rule (for low base scores)
    # This ensures low scores get lifted based on top modality importance
    final_score = base_score * 0.7 + top_weight * 0.3
    return round(final_score, 3)


def compute_audio_search_score(modality_scores: Dict[str, float]) -> float:
    """
    Computes a final score specifically for audio-based searches.
    Priority is given to audio similarity, then to text similarity.
    
    Args:
        modality_scores: Dictionary with 'audio' and optionally 'text' scores
        
    Returns:
        Final score (float, rounded to 3 decimal places)
    """
    audio_score = modality_scores.get('audio', 0.0)
    text_score = modality_scores.get('text', 0.0)
    
    if audio_score >= 0.75:
        return round(audio_score, 3)
    
    final_score = (audio_score * 1.0 + text_score * 0.4) / 1.4
    return round(final_score, 3)

