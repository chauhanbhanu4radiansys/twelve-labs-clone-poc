"""
Analysis functionality for video content
Supports query intent detection and LLM-based analysis
"""
import json
from typing import Dict, Optional, List, Any
from functools import wraps
from time import sleep

try:
    import openai
except ImportError:
    openai = None


def retry_on_network_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry functions on network errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if it's a network-related error
                    if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'axios', 'http', 'request']):
                        last_exception = e
                        retries += 1
                        if retries < max_retries:
                            sleep(current_delay)
                            current_delay *= backoff
                            continue
                    # If it's not a network error, raise immediately
                    raise
            # If we exhausted retries, raise the last exception
            raise last_exception
        return wrapper
    return decorator


def get_query_intent(prompt: str, openai_client=None) -> str:
    """
    Classifies the user's prompt as 'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST' using an LLM.
    
    Args:
        prompt: User's query/prompt
        openai_client: OpenAI client instance (optional)
        
    Returns:
        'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST'
    """
    if not openai_client or openai is None:
        return "SPECIFIC_QUESTION"  # Default fallback if OpenAI isn't configured
    
    system_prompt = """You are an intent detection agent. Classify the user's query about a video transcript as either 'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST'.
- SPECIFIC_QUESTION is for queries asking about a particular detail, fact, or event.
- HOLISTIC_REQUEST is for queries asking for a summary, highlights, or an overview of the entire video.
Your response must be ONLY 'SPECIFIC_QUESTION' or 'HOLISTIC_REQUEST'."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
            timeout=15.0
        )
        intent = response.choices[0].message.content.strip()
        # Validate the response from the LLM
        if intent in ["SPECIFIC_QUESTION", "HOLISTIC_REQUEST"]:
            return intent
        return "SPECIFIC_QUESTION"  # Fallback on any unexpected response
    except Exception:
        return "SPECIFIC_QUESTION"  # Fallback on API error


def get_llm_analysis_json(
    prompt: str,
    context: str,
    video_duration: float,
    openai_client=None
) -> Optional[Dict[str, Any]]:
    """
    Sends a prompt and context to the LLM and requests a structured JSON response.
    
    Args:
        prompt: User's analysis request
        context: Video transcript context (formatted with timestamps)
        video_duration: Total video duration in seconds
        openai_client: OpenAI client instance
        
    Returns:
        Dictionary with 'analysis_text' and 'clips' keys, or None if error
    """
    if not openai_client or openai is None:
        return None
    
    system_prompt = f"""You are a video content analyst. Your goal is to identify a diverse set of distinct moments from different parts of the video based on the user's request.
The total duration of the video is {int(video_duration)} seconds.

You will receive a video transcript context, with each line prefixed by its timestamp (e.g., [start-end s]). Provide a comprehensive, detailed response based *only* on the provided context. If the answer is not in the context, state that clearly.

Your response MUST be a JSON object with two keys:
1. "analysis_text": A string containing your full, detailed analysis.
2. "clips": A list of JSON objects representing mentioned clips. Each object must have "description" (string) and "start_time" (integer in seconds).

IMPORTANT RULES for the "clips" list:
- Each clip in the 'clips' list MUST have a unique 'start_time'.
- The 'start_time' for any clip MUST NOT exceed the video duration of {int(video_duration)} seconds.
- Base your 'start_time' on the timestamps provided in the context (e.g., [15.5-20.0s]).
- If multiple relevant events occur at the same start time, consolidate them into a single, more descriptive clip entry.
- If no specific clips are mentioned or found, provide an empty list.

Example clip: {{"description": "The speaker discusses Q3 earnings and future projections.", "start_time": 125}}"""
    
    user_message = f"""Context from Video:
---
{context}
---

User Request: {prompt}

Please provide your analysis in the required JSON format based on the context above."""
    
    try:
        @retry_on_network_error(max_retries=3, delay=2.0)
        def call_openai_api():
            return openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.5,
                max_tokens=2000,
                response_format={"type": "json_object"},
                timeout=60.0
            )
        
        response = call_openai_api()
        analysis_result = response.choices[0].message.content
        return json.loads(analysis_result)
    except Exception as e:
        print(f"OpenAI analysis failed: {e}")
        return None


def format_transcript_context(transcript_segments: List[Dict]) -> str:
    """
    Formats transcript segments into a context string with timestamps.
    
    Args:
        transcript_segments: List of transcript segment dictionaries with 'start', 'end', 'text' keys
        
    Returns:
        Formatted context string
    """
    if not transcript_segments:
        return ""
    
    context_parts = []
    for seg in transcript_segments:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        text = seg.get('text', '').strip()
        
        if text:
            context_parts.append(f"[{start:.1f}-{end:.1f}s] {text}")
    
    return "\n".join(context_parts)


def analyze_video(
    prompt: str,
    video_doc_id: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    video_duration: Optional[float] = None,
    search_results: Optional[List[Any]] = None,
    embedding_model=None,
    device: str = "cpu",
    openai_client=None,
    text_index=None,
    desc_index=None
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a video using either holistic (full transcript) or RAG-based (search results) approach.
    
    Args:
        prompt: User's analysis request
        video_doc_id: Optional video document ID for filtering search results
        transcript_segments: Full transcript segments (for holistic analysis)
        video_duration: Video duration in seconds
        search_results: Pre-computed search results (for RAG-based analysis)
        embedding_model: ImageBind model (for generating query embeddings if needed)
        device: Device to run model on
        openai_client: OpenAI client instance
        text_index: Pinecone text index (for RAG-based analysis)
        desc_index: Pinecone description index (for RAG-based analysis)
        
    Returns:
        Dictionary with 'analysis_text' and 'clips' keys, or None if error
    """
    if not openai_client or openai is None:
        print("OpenAI client is not configured")
        return None
    
    if not video_duration:
        print("Video duration is required")
        return None
    
    # Detect query intent
    intent = get_query_intent(prompt, openai_client)
    
    if intent == "HOLISTIC_REQUEST":
        # Use full transcript for holistic analysis
        if not transcript_segments:
            print("Transcript segments required for holistic analysis")
            return None
        
        context = format_transcript_context(transcript_segments)
        return get_llm_analysis_json(prompt, context, video_duration, openai_client)
    
    else:  # SPECIFIC_QUESTION - Use RAG approach
        # If search results provided, use them
        if search_results:
            unique_clips_context = {}
            for match in sorted(search_results, key=lambda x: x.score, reverse=True):
                meta = match.metadata
                start = meta.get('start_time', 0.0)
                end = meta.get('end_time', 0.0)
                text = meta.get("transcript") or meta.get("description", "")
                if text:
                    # Use int(start) as key to avoid duplicates from very close timestamps
                    if int(start) not in unique_clips_context:
                        unique_clips_context[int(start)] = f"[{start:.1f}-{end:.1f}s] {text}"
            
            context = "\n".join(unique_clips_context.values())
            if context:
                return get_llm_analysis_json(prompt, context, video_duration, openai_client)
        
        # If no search results but we have transcript, fallback to full transcript
        if transcript_segments:
            context = format_transcript_context(transcript_segments)
            return get_llm_analysis_json(prompt, context, video_duration, openai_client)
        
        print("No context available for analysis")
        return None

