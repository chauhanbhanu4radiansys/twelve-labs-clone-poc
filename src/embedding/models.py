"""
Model loading for ImageBind, Whisper, and BLIP
"""
import os
import torch
import logging
import warnings
from typing import Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='imagebind')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*torch.utils._pytree._register_pytree_node is deprecated.*')
warnings.filterwarnings('ignore', message='.*Torchaudio.*backend.*')

# Configure ImageBind logging
try:
    logging.getLogger('imagebind.data').disabled = True
    logging.getLogger('imagebind').disabled = True
    for logger_name in ['imagebind', 'imagebind.data']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers = []
        logger.propagate = False
except Exception:
    pass

try:
    import whisper
except ImportError:
    whisper = None
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    print("Warning: Transformers not available. Install with: pip install transformers")

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def load_imagebind_model(checkpoint_path: Optional[str] = None) -> Tuple[any, str]:
    """
    Loads the ImageBind model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses default location.
        
    Returns:
        Tuple of (model, device)
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if checkpoint_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoint_path = os.path.join(project_root, ".checkpoints", "imagebind_huge.pth")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at '{checkpoint_path}'. Downloading model weights...")
            model = imagebind_model.imagebind_huge(pretrained=True)
        else:
            print(f"Loading ImageBind model from '{checkpoint_path}'...")
            model = imagebind_model.imagebind_huge(pretrained=False)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        
        try:
            model.to(device)
            if device.startswith('cuda'):
                next(model.parameters()).device
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if device.startswith('cuda'):
                print(f"CUDA OOM. Falling back to CPU. Error: {e}")
                device = "cpu"
                torch.cuda.empty_cache()
                model.to(device)
            else:
                raise
        
        print(f"ImageBind model loaded on {device}")
        return model, device
        
    except Exception as e:
        print(f"Failed to load ImageBind model: {e}")
        if device.startswith('cuda'):
            try:
                print("Attempting to load on CPU...")
                device = "cpu"
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model = imagebind_model.imagebind_huge(pretrained=False)
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'], strict=False)
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    model = imagebind_model.imagebind_huge(pretrained=True)
                model.eval()
                model.to(device)
                print("Model loaded successfully on CPU.")
                return model, device
            except Exception as cpu_error:
                print(f"Failed to load model on CPU: {cpu_error}")
                raise
        else:
            raise


def load_whisper_model(model_name: str = "base"):
    """
    Loads a Whisper model.
    
    Args:
        model_name: Whisper model name (tiny, base, small, medium, large)
        
    Returns:
        Whisper model or None if not available
    """
    if whisper is None:
        return None
    
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Whisper model loaded successfully")
        return model
    except Exception as e:
        print(f"Could not load Whisper model: {e}")
        return None


def load_captioning_model() -> Tuple[Optional[any], Optional[any]]:
    """
    Loads the BLIP image captioning model and processor.
    
    Returns:
        Tuple of (processor, model) or (None, None) if not available
    """
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        return None, None
    
    try:
        print("Loading BLIP captioning model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        if torch.cuda.is_available():
            model.to("cuda:0")
        
        print("BLIP model loaded successfully")
        return processor, model
    except Exception as e:
        print(f"Could not load captioning model: {e}")
        return None, None

