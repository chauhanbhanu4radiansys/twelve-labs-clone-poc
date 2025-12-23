"""
Model loading for ImageBind, Whisper, and BLIP
"""
import os
import sys
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
warnings.filterwarnings('ignore', message='.*Large gap between audio.*')

# Configure thread-safe logging to prevent "I/O operation on closed file" errors
class NullHandler(logging.Handler):
    """A handler that does nothing, preventing logging errors in threads."""
    def emit(self, record):
        pass
    
    def handle(self, record):
        pass
    
    def flush(self):
        pass

# Configure ImageBind logging to be thread-safe
try:
    # Disable all ImageBind loggers completely
    for logger_name in ['imagebind', 'imagebind.data', 'imagebind.models']:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        # Remove all existing handlers that might be closed
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        # Add a null handler to prevent errors
        null_handler = NullHandler()
        logger.addHandler(null_handler)
        logger.propagate = False
    
    # Patch ALL StreamHandler instances to handle closed file errors gracefully
    # This must be done before any handlers are created
    original_emit = logging.StreamHandler.emit
    def safe_emit(self, record):
        try:
            # Check if stream exists and is not closed
            if not hasattr(self, 'stream') or self.stream is None:
                return
            if hasattr(self.stream, 'closed') and self.stream.closed:
                return
            # Check if stream is writable
            if hasattr(self.stream, 'writable'):
                try:
                    if not self.stream.writable():
                        return
                except (ValueError, OSError, AttributeError):
                    return
            # Try to write, catch any I/O errors
            original_emit(self, record)
        except (ValueError, OSError, AttributeError, RuntimeError) as e:
            # Ignore all I/O and stream-related errors - logging failures shouldn't crash the app
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                "closed file", "i/o operation", "bad file descriptor", 
                "broken pipe", "connection", "stream"
            ]):
                return  # Silently ignore
            # For any other errors, also ignore to prevent crashes
            return
    logging.StreamHandler.emit = safe_emit
    
    # Patch existing handlers in root logger and all loggers to use safe_emit
    # This ensures handlers created before the patch are also protected
    import types
    def patch_existing_handlers():
        """Patch all existing StreamHandlers to use safe_emit"""
        for logger_name in ['', 'imagebind', 'imagebind.data', 'imagebind.models']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    # Create a bound method that uses safe_emit
                    def make_safe_emit(h):
                        def safe_emit_bound(record):
                            return safe_emit(h, record)
                        return safe_emit_bound
                    handler.emit = make_safe_emit(handler)
    
    patch_existing_handlers()
    
    # Also ensure ImageBind loggers don't propagate to root logger
    # This prevents root logger handlers from trying to write to closed streams
    for logger_name in ['imagebind', 'imagebind.data', 'imagebind.models']:
        logger = logging.getLogger(logger_name)
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


def load_captioning_model(device: Optional[str] = None) -> Tuple[Optional[any], Optional[any]]:
    """
    Loads the BLIP image captioning model and processor.
    
    Args:
        device: Device to load model on (e.g., "cuda:0" or "cpu"). 
                If None, auto-detects: uses CUDA if available, else CPU.
    
    Returns:
        Tuple of (processor, model) or (None, None) if not available
    """
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        return None, None
    
    try:
        print("Loading BLIP captioning model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Determine device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Move model to device
        try:
            model.to(device)
            if device.startswith('cuda'):
                # Verify model is actually on GPU
                actual_device = next(model.parameters()).device
                if actual_device.type != 'cuda':
                    print(f"Warning: BLIP model requested GPU but is on {actual_device}")
        except Exception as e:
            print(f"Warning: Could not move BLIP model to {device}: {e}")
            # Try CPU as fallback
            if device.startswith('cuda'):
                print("Falling back to CPU for BLIP model...")
                device = "cpu"
                model.to(device)
        
        print(f"BLIP model loaded successfully on {device}")
        return processor, model
    except Exception as e:
        print(f"Could not load captioning model: {e}")
        return None, None

