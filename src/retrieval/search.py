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
    
    Args:
        query_text: Text query string
        video_index: Pinecone video index
        audio_index: Pinecone audio index
        text_index: Pinecone text index
        desc_index: Pinecone description index
        embedding_model: ImageBind model for generating embeddings
        caption_processor: BLIP processor (not used for text search)
        caption_model: BLIP model (not used for text search)
        device: Device to run model on ('cuda:0' or 'cpu')
        top_k: Number of results per index
        filter_dict: Optional filter for metadata (e.g., {"video_doc_id": "..."})
        
    Returns:
        List of search results sorted by score (descending)
    """
    # Generate text embedding
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
    
    # Prepare search tasks for all indexes
    search_tasks = {
        video_index: query_vector,
        audio_index: query_vector,
        text_index: query_vector,
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
    
    # Compute final scores and create result list
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        source_details = [source for source in sorted(scores_dict.keys())]
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
    # Load image
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return []
    
    # Generate image embedding
    query_embedding_tensor, _, _ = get_batch_embeddings(
        pil_images=[pil_image],
        audio_paths=[],
        texts=[],
        device=device,
        model=embedding_model
    )
    
    search_tasks = {}
    if query_embedding_tensor.nelement() > 0:
        search_tasks[video_index] = query_embedding_tensor[0].cpu().numpy().tolist()
    
    # Generate description and its embedding
    if caption_processor and caption_model:
        desc = generate_scene_descriptions([pil_image], caption_processor, caption_model, device)[0]
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
    
    if not search_tasks:
        return []
    
    # Query indexes in parallel
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
    
    # Compute final scores
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        final_score = compute_final_score(scores_dict)
        best_match.score = final_score
        source_details = [source for source in sorted(scores_dict.keys())]
        best_match.metadata['source'] = ', '.join(source_details)
        final_results.append(best_match)
    
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
    
    # Compute final scores using audio-specific scoring
    final_results = []
    for item in merged_results.values():
        best_match = item['match']
        scores_dict = item['scores']
        best_match.score = compute_audio_search_score(scores_dict)
        final_results.append(best_match)
    
    return sorted(final_results, key=lambda x: x.score, reverse=True)


def search_videos(
    query: Optional[str] = None,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
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
    gap_seconds: int = 3
) -> List[Any]:
    """
    Unified search function that handles text, image, or audio queries.
    
    Args:
        query: Text query string (optional)
        image_path: Path to image file (optional)
        audio_path: Path to audio file (optional)
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
    results = []
    
    # Priority: audio > image > text
    if audio_path:
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
    elif image_path:
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
    elif query:
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
    
    # Merge overlapping clips if requested
    if merge_clips and results:
        results = merge_overlapping_clips(results, gap_seconds=gap_seconds)
    
    return results

