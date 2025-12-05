"""
Search functionality for video retrieval
Supports text, image, and audio queries
"""
import os
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import torch

from ..embedding.embeddings import get_batch_embeddings, generate_scene_descriptions
from ..embedding.models import load_whisper_model
from .scoring import compute_final_score, compute_audio_search_score
from .utils import merge_overlapping_clips


def query_index(index, query_vec: List[float], top_k: int = 8, filter_dict: Optional[Dict] = None):
    """
    Queries a Pinecone index with a query vector.
    
    Args:
        index: Pinecone index object
        query_vec: Query vector (list of floats)
        top_k: Number of results to return
        filter_dict: Optional filter dictionary for metadata filtering
        
    Returns:
        Query results from Pinecone
    """
    if filter_dict:
        return index.query(vector=query_vec, top_k=top_k, include_metadata=True, filter=filter_dict)
    return index.query(vector=query_vec, top_k=top_k, include_metadata=True)


def perform_text_search(
    query_text: str,
    video_index,
    audio_index,
    text_index,
    desc_index,
    embedding_model,
    caption_processor,
    caption_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Performs a text-based search across all modalities.
    
    IMPORTANT: This function queries the text_index which contains embeddings from transcript text.
    For queries asking about what speakers say (e.g., "where does a speaker say X"),
    the text_index will provide the most relevant matches as it contains transcript embeddings.
    The scoring function prioritizes text index results when they have high relevance scores.
    
    Args:
        query_text: Text query string (e.g., "where does a speaker say X")
        video_index: Pinecone video index (visual embeddings)
        audio_index: Pinecone audio index (audio embeddings)
        text_index: Pinecone text index (transcript text embeddings) - KEY for transcript queries
        desc_index: Pinecone description index (scene description embeddings)
        embedding_model: ImageBind model for generating embeddings
        caption_processor: BLIP processor (not used for text search)
        caption_model: BLIP model (not used for text search)
        device: Device to run model on ('cuda:0' or 'cpu')
        top_k: Number of results per index
        filter_dict: Optional filter for metadata (e.g., {"video_doc_id": "..."})
        
    Returns:
        List of search results sorted by score (descending)
        Results from text_index are prioritized when they have high relevance (score >= 0.65)
    """
    # Generate text embedding from query
    _, _, query_embedding_tensor = get_batch_embeddings(
        pil_images=[],
        audio_paths=[],
        texts=[query_text],
        device=device,
        model=embedding_model
    )
    
    if query_embedding_tensor.nelement() == 0:
        return []
    
    query_vector = query_embedding_tensor[0].cpu().numpy().tolist()
    
    # Query all indexes in parallel with the same text embedding
    # text_index contains transcript embeddings, so it's crucial for transcript-related queries
    search_tasks = {
        video_index: query_vector,
        audio_index: query_vector,
        text_index: query_vector,  # This index contains transcript text embeddings
        desc_index: query_vector
    }
    
    # Query all indexes in parallel
    results_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(query_index, index, vec, top_k, filter_dict): index
            for index, vec in search_tasks.items()
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results_map[index] = future.result(timeout=30)
            except Exception as e:
                print(f"Error querying index: {e}")
                results_map[index] = None
    
    # Process and merge results
    merged_results = {}
    
    index_to_name = {
        video_index: 'video',
        audio_index: 'audio',
        text_index: 'text',
        desc_index: 'desc'
    }
    
    def process_results(results, source_name):
        if not results or not hasattr(results, 'matches'):
            return
        for match in results.matches:
            scene_uuid = match.metadata.get("scene_uuid")
            if not scene_uuid:
                continue
            
            if scene_uuid not in merged_results:
                merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
            else:
                merged_results[scene_uuid]['scores'][source_name] = match.score
                if match.score > merged_results[scene_uuid]['match'].score:
                    merged_results[scene_uuid]['match'] = match
    
    for index, results in results_map.items():
        source_name = index_to_name.get(index)
        if source_name and results:
            process_results(results, source_name)
    
    # Compute final scores and create result list (matching app-code-ref.py logic exactly)
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        # Match app-code-ref.py: sort sources alphabetically (same as sorted(scores_dict.items()) which sorts by key)
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def perform_image_search(
    image_path: str,
    video_index,
    desc_index,
    embedding_model,
    caption_processor,
    caption_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Performs an image-based search using video and description indexes.
    
    Args:
        image_path: Path to image file
        video_index: Pinecone video index
        desc_index: Pinecone description index
        embedding_model: ImageBind model for generating embeddings
        caption_processor: BLIP processor for generating descriptions
        caption_model: BLIP model for generating descriptions
        device: Device to run model on ('cuda:0' or 'cpu')
        top_k: Number of results per index
        filter_dict: Optional filter for metadata
        
    Returns:
        List of search results sorted by score (descending)
    """
    print(f"[DEBUG] Starting image search with image_path: {image_path}")
    
    # Load image
    try:
        pil_image = Image.open(image_path)
        print(f"[DEBUG] Image loaded successfully. Size: {pil_image.size}, Mode: {pil_image.mode}")
    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # Generate image embedding
    print(f"[DEBUG] Generating image embedding...")
    query_embedding_tensor, _, _ = get_batch_embeddings(
        pil_images=[pil_image],
        audio_paths=[],
        texts=[],
        device=device,
        model=embedding_model
    )
    
    print(f"[DEBUG] Image embedding generated. Shape: {query_embedding_tensor.shape if query_embedding_tensor.nelement() > 0 else 'empty'}")
    
    search_tasks = {}
    if query_embedding_tensor.nelement() > 0:
        search_tasks[video_index] = query_embedding_tensor[0].cpu().numpy().tolist()
        print(f"[DEBUG] Added video_index to search_tasks. Vector length: {len(search_tasks[video_index])}")
    else:
        print(f"[DEBUG] WARNING: Image embedding is empty!")
    
    # Generate description and its embedding
    if caption_processor and caption_model:
        print(f"[DEBUG] Generating description...")
        desc = generate_scene_descriptions([pil_image], caption_processor, caption_model, device)[0]
        print(f"[DEBUG] Description generated: {desc[:100] if desc else 'EMPTY'}...")
        if desc:
            _, _, desc_embedding_tensor = get_batch_embeddings(
                pil_images=[],
                audio_paths=[],
                texts=[desc],
                device=device,
                model=embedding_model
            )
            if desc_embedding_tensor.nelement() > 0:
                search_tasks[desc_index] = desc_embedding_tensor[0].cpu().numpy().tolist()
                print(f"[DEBUG] Added desc_index to search_tasks. Vector length: {len(search_tasks[desc_index])}")
            else:
                print(f"[DEBUG] WARNING: Description embedding is empty!")
        else:
            print(f"[DEBUG] WARNING: Description is empty!")
    else:
        print(f"[DEBUG] WARNING: caption_processor or caption_model is None!")
    
    print(f"[DEBUG] Total search_tasks: {len(search_tasks)}")
    
    if not search_tasks:
        print(f"[DEBUG] ERROR: No search_tasks created! Returning empty results.")
        return []
    
    # Query indexes in parallel
    print(f"[DEBUG] Querying Pinecone indexes...")
    results_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(query_index, index, vec, top_k, filter_dict): index
            for index, vec in search_tasks.items()
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results = future.result(timeout=30)
                results_map[index] = results
                if results and hasattr(results, 'matches'):
                    print(f"[DEBUG] Index {index} returned {len(results.matches)} matches")
                else:
                    print(f"[DEBUG] Index {index} returned no matches or invalid result")
            except Exception as e:
                print(f"Error querying index: {e}")
                import traceback
                traceback.print_exc()
                results_map[index] = None
    
    print(f"[DEBUG] Results map has {len(results_map)} entries")
    
    # Process and merge results
    merged_results = {}
    
    index_to_name = {
        video_index: 'video',
        desc_index: 'desc'
    }
    
    def process_results(results, source_name):
        if not results or not hasattr(results, 'matches'):
            print(f"[DEBUG] Skipping results for {source_name}: no matches or invalid")
            return
        print(f"[DEBUG] Processing {len(results.matches)} matches from {source_name}")
        for match in results.matches:
            scene_uuid = match.metadata.get("scene_uuid")
            if not scene_uuid:
                print(f"[DEBUG] WARNING: Match missing scene_uuid, skipping")
                continue
            
            if scene_uuid not in merged_results:
                merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
            else:
                merged_results[scene_uuid]['scores'][source_name] = match.score
                if match.score > merged_results[scene_uuid]['match'].score:
                    merged_results[scene_uuid]['match'] = match
    
    for index, results in results_map.items():
        source_name = index_to_name.get(index)
        if source_name and results:
            process_results(results, source_name)
    
    print(f"[DEBUG] Merged results: {len(merged_results)} unique scenes")
    
    # Compute final scores (matching app-code-ref.py logic exactly)
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        # Match app-code-ref.py: sort sources alphabetically
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    print(f"[DEBUG] Final results count: {len(final_results)}")
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def perform_audio_search(
    audio_path: str,
    audio_index,
    text_index,
    embedding_model,
    whisper_model,
    device: str,
    top_k: int = 8,
    filter_dict: Optional[Dict] = None
) -> List[Any]:
    """
    Performs an audio-based search using audio and text (transcript) indexes.
    
    Args:
        audio_path: Path to audio file
        audio_index: Pinecone audio index
        text_index: Pinecone text index
        embedding_model: ImageBind model for generating embeddings
        whisper_model: Whisper model for transcription
        device: Device to run model on ('cuda:0' or 'cpu')
        top_k: Number of results per index
        filter_dict: Optional filter for metadata
        
    Returns:
        List of search results sorted by score (descending)
    """
    search_tasks = {}
    
    # 1. Generate audio embedding
    _, audio_embedding_tensor, _ = get_batch_embeddings(
        pil_images=[],
        audio_paths=[audio_path],
        texts=[],
        device=device,
        model=embedding_model
    )
    
    if audio_embedding_tensor.nelement() > 0:
        search_tasks[audio_index] = audio_embedding_tensor[0].cpu().numpy().tolist()
    
    # 2. Transcribe audio and generate text embedding
    transcript = None
    if whisper_model and os.path.exists(audio_path):
        try:
            result = whisper_model.transcribe(
                audio_path,
                fp16=torch.cuda.is_available() if torch.cuda.is_available() else False,
                task='translate'
            )
            transcript = result.get("text", "").strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
    
    if transcript:
        _, _, text_embedding_tensor = get_batch_embeddings(
            pil_images=[],
            audio_paths=[],
            texts=[transcript],
            device=device,
            model=embedding_model
        )
        if text_embedding_tensor.nelement() > 0:
            search_tasks[text_index] = text_embedding_tensor[0].cpu().numpy().tolist()
    
    if not search_tasks:
        return []
    
    # 3. Query Pinecone indexes in parallel
    results_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(query_index, index, vec, top_k, filter_dict): index
            for index, vec in search_tasks.items()
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results_map[index] = future.result(timeout=30)
            except Exception as e:
                print(f"Error querying index: {e}")
                results_map[index] = None
    
    # 4. Process and score results
    merged_results = {}
    
    index_to_name = {
        audio_index: 'audio',
        text_index: 'text'
    }
    
    for index, results in results_map.items():
        source_name = index_to_name.get(index)
        if source_name and results and hasattr(results, 'matches'):
            for match in results.matches:
                scene_uuid = match.metadata.get("scene_uuid")
                if not scene_uuid:
                    continue
                
                if scene_uuid not in merged_results:
                    merged_results[scene_uuid] = {'match': match, 'scores': {source_name: match.score}}
                else:
                    merged_results[scene_uuid]['scores'][source_name] = match.score
                    if match.score > merged_results[scene_uuid]['match'].score:
                        merged_results[scene_uuid]['match'] = match
    
    # Compute final scores using audio-specific scoring (matching app-code-ref.py logic)
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        best_match.score = compute_audio_search_score(scores_dict)
        # Set source metadata (matching app-code-ref.py format)
        source_details = [source for source, score in sorted(scores_dict.items())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def search_videos(
    search_type: str,
    query: str = None,
    video_index=None,
    audio_index=None,
    text_index=None,
    desc_index=None,
    embedding_model=None,
    caption_processor=None,
    caption_model=None,
    whisper_model=None,
    device: str = "cpu",
    top_k: int = 8,
    filter_dict: Optional[Dict] = None,
    merge_clips: bool = True,
    gap_seconds: int = 3,
    **kwargs  # For backward compatibility with 'search_query'
) -> List[Any]:
    """
    Unified search function that handles text, image, or audio queries.
    
    Args:
        search_type: Type of search - 'text', 'image', or 'audio'
        query: The search query - text string for text search, URL for image/audio search (preferred)
        **kwargs: For backward compatibility, accepts 'search_query' as alternative to 'query'
        video_index: Pinecone video index
        audio_index: Pinecone audio index
        text_index: Pinecone text index
        desc_index: Pinecone description index
        embedding_model: ImageBind model
        caption_processor: BLIP processor
        caption_model: BLIP model
        whisper_model: Whisper model
        device: Device to run models on
        top_k: Number of results per index
        filter_dict: Optional metadata filter
        merge_clips: Whether to merge overlapping clips
        gap_seconds: Gap in seconds for merging clips
        
    Returns:
        List of search results, optionally merged
    """
    import tempfile
    import requests
    
    # Support both 'query' and 'search_query' for backward compatibility
    if query is None:
        query = kwargs.get('search_query')
    if query is None:
        raise ValueError("Missing required parameter: query (or search_query for backward compatibility)")
    
    results = []
    temp_file_path = None
    
    try:
        if search_type == 'audio':
            # Handle both URLs (S3 pre-signed URLs, HTTP/HTTPS) and local file paths
            if query.startswith('http://') or query.startswith('https://'):
                # Download audio from URL (supports S3 pre-signed URLs and regular HTTP/HTTPS URLs)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', delete_on_close=False)
                temp_file_path = temp_file.name
                temp_file.close()
                
                # Download with streaming for large files (better for S3)
                # Use longer timeout for S3 pre-signed URLs which may be slower
                try:
                    print(f"Downloading audio from URL: {query[:100]}...")
                    response = requests.get(query, timeout=120, stream=True)
                    response.raise_for_status()
                    
                    # Stream download for better memory efficiency (important for large S3 files)
                    with open(temp_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    print(f"Audio downloaded successfully to: {temp_file_path}")
                    audio_path = temp_file_path
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading audio from URL: {e}")
                    # Clean up temp file if download failed
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except OSError:
                            pass
                    raise ValueError(f"Failed to download audio from URL: {str(e)}")
            else:
                # Local file path (for testing or mounted files)
                if not os.path.exists(query):
                    raise FileNotFoundError(f"Audio file not found: {query}")
                audio_path = query
            
            results = perform_audio_search(
                audio_path=audio_path,
                audio_index=audio_index,
                text_index=text_index,
                embedding_model=embedding_model,
                whisper_model=whisper_model,
                device=device,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
        elif search_type == 'image':
            # Handle both URLs (S3 pre-signed URLs, HTTP/HTTPS) and local file paths
            if query.startswith('http://') or query.startswith('https://'):
                # Download image from URL (supports S3 pre-signed URLs and regular HTTP/HTTPS URLs)
                # Determine file extension from URL (handle query parameters in S3 URLs)
                url_path = query.split('?')[0]
                ext = os.path.splitext(url_path)[1]
                # Default to .png if extension not found
                if not ext or ext == '':
                    ext = '.png'
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, delete_on_close=False)
                temp_file_path = temp_file.name
                temp_file.close()
                
                # Download with streaming for large files (better for S3)
                # Use longer timeout for S3 pre-signed URLs which may be slower
                try:
                    print(f"Downloading image from URL: {query[:100]}...")
                    response = requests.get(query, timeout=120, stream=True)
                    response.raise_for_status()
                    
                    # Stream download for better memory efficiency (important for large S3 files)
                    with open(temp_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    print(f"Image downloaded successfully to: {temp_file_path}")
                    image_path = temp_file_path
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading image from URL: {e}")
                    # Clean up temp file if download failed
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except OSError:
                            pass
                    raise ValueError(f"Failed to download image from URL: {str(e)}")
            else:
                # Local file path (for testing or mounted files)
                if not os.path.exists(query):
                    raise FileNotFoundError(f"Image file not found: {query}")
                image_path = query
            
            results = perform_image_search(
                image_path=image_path,
                video_index=video_index,
                desc_index=desc_index,
                embedding_model=embedding_model,
                caption_processor=caption_processor,
                caption_model=caption_model,
                device=device,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
        elif search_type == 'text':
            results = perform_text_search(
                query_text=query,
                video_index=video_index,
                audio_index=audio_index,
                text_index=text_index,
                desc_index=desc_index,
                embedding_model=embedding_model,
                caption_processor=caption_processor,
                caption_model=caption_model,
                device=device,
                top_k=top_k,
                filter_dict=filter_dict
            )
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Must be 'text', 'image', or 'audio'.")
        
        # Merge overlapping clips if requested
        if merge_clips and results:
            results = merge_overlapping_clips(results, gap_seconds=gap_seconds)
        
    finally:
        # Clean up temporary file if created (for both image and audio downloads from URLs)
        # This ensures cleanup even if an error occurs during processing
        # Important for production S3 pre-signed URLs to prevent disk space issues
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                print(f"Warning: Could not clean up temporary file {temp_file_path}: {e}")
    
    return results

