"""
Embedding generation using ImageBind
"""
import os
import tempfile
import torch
import logging
import warnings
from typing import List, Tuple
from PIL import Image

# Suppress ImageBind logging warnings before importing
warnings.filterwarnings('ignore', message='.*Large gap between audio.*')
logging.getLogger('imagebind.data').setLevel(logging.ERROR)
logging.getLogger('imagebind').setLevel(logging.ERROR)

from imagebind import data
from imagebind.models.imagebind_model import ModalityType


def get_batch_embeddings(
    pil_images: List[Image.Image],
    audio_paths: List[str],
    texts: List[str],
    device: str,
    model
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates embeddings for a batch of images, audio clips, and texts."""
    image_embeddings = torch.empty((0,))
    audio_embeddings = torch.empty((0,))
    text_embeddings = torch.empty((0,))

    # Process images if any
    if pil_images:
        temp_image_paths = []
        try:
            for pil_image in pil_images:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    pil_image.save(temp_file, format='PNG')
                    temp_image_paths.append(temp_file.name)
            
            vision_inputs = {ModalityType.VISION: data.load_and_transform_vision_data(temp_image_paths, device)}
            with torch.no_grad():
                image_embeddings = model(vision_inputs)[ModalityType.VISION]
        finally:
            for path in temp_image_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # Process audio if any
    if audio_paths:
        try:
            valid_audio_paths = [path for path in audio_paths if path and os.path.exists(path)]
            if valid_audio_paths:
                # Suppress ImageBind logging to prevent thread-safety issues
                imagebind_logger = logging.getLogger('imagebind.data')
                imagebind_logger.setLevel(logging.CRITICAL)
                
                try:
                    audio_inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(valid_audio_paths, device)}
                    with torch.no_grad():
                        audio_embeddings = model(audio_inputs)[ModalityType.AUDIO]
                except ValueError as ve:
                    # Suppress logging errors - they don't affect functionality
                    error_str = str(ve)
                    if "I/O operation on closed file" in error_str or "logging" in error_str.lower():
                        # Logging error, ignore it and continue
                        pass
                    else:
                        # Real ValueError, re-raise
                        raise
        except (ValueError, OSError, Exception) as e:
            error_msg = str(e)
            # Ignore logging errors and I/O errors on closed files
            if "I/O operation on closed file" not in error_msg and "logging" not in error_msg.lower():
                print(f"Error getting audio embeddings: {e}")
            
    # Process text if any
    if texts:
        try:
            text_inputs = {ModalityType.TEXT: data.load_and_transform_text(texts, device)}
            with torch.no_grad():
                text_embeddings = model(text_inputs)[ModalityType.TEXT]
        except Exception as e:
            print(f"Error getting text embeddings: {e}")

    return image_embeddings, audio_embeddings, text_embeddings


def generate_scene_descriptions(
    pil_images: List[Image.Image], 
    processor, 
    model, 
    device: str
) -> List[str]:
    """Generates text descriptions for a batch of image frames using BLIP."""
    if not processor or not model or not pil_images:
        return [""] * len(pil_images)
    
    try:
        model_device = next(model.parameters()).device
        inputs = processor(images=pil_images, return_tensors="pt").to(model_device)
        
        generated_ids = model.generate(**inputs, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [caption.strip() for caption in generated_captions]
    except Exception as e:
        print(f"Error during batch caption generation: {e}")
        return [""] * len(pil_images)

