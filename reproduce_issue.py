
import logging
import sys
import os

# Mocking the structure to match the issue
class MockStream:
    def __init__(self):
        self.closed = True
    
    def write(self, msg):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        print(msg)
    
    def flush(self):
        pass

# Mocking the original emit
def original_emit(self, record):
    if hasattr(self, 'stream'):
        self.stream.write(self.format(record))

# The safe_emit function from models.py (copy-pasted)
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
            print(f"Caught expected error: {e}")
            return  # Silently ignore
        # For any other errors, also ignore to prevent crashes
        print(f"Caught unexpected error: {e}")
        return

# Test the safe_emit function
def test_safe_emit():
    print("Testing safe_emit...")
    handler = logging.StreamHandler()
    handler.stream = MockStream()
    
    # Patch it
    handler.emit = lambda record: safe_emit(handler, record)
    
    record = logging.LogRecord("name", logging.INFO, "pathname", 1, "msg", (), None)
    
    try:
        handler.emit(record)
        print("safe_emit handled the error successfully.")
    except Exception as e:
        print(f"safe_emit FAILED to handle error: {e}")

# Mocking embeddings.py logic
class ModalityType:
    AUDIO = "audio"

def test_embeddings_logic():
    print("\nTesting embeddings.py logic...")
    try:
        # Simulate the block in embeddings.py
        try:
            # Simulate raising the error
            raise ValueError("I/O operation on closed file")
        except ValueError as ve:
            # Suppress logging errors - they don't affect functionality
            error_str = str(ve)
            if "I/O operation on closed file" in error_str or "logging" in error_str.lower():
                # Logging error, ignore it and continue
                print("embeddings.py logic handled the error successfully.")
                pass
            else:
                # Real ValueError, re-raise
                raise
    except Exception as e:
        print(f"embeddings.py logic FAILED to handle error: {e}")

if __name__ == "__main__":
    test_safe_emit()
    test_embeddings_logic()
