"""
Configuration loaded from environment variables
"""
import os

# ============================================================================
# Pinecone Configuration - Loaded from Environment Variables
# ============================================================================
PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY", ""),
    "video_index_name": os.getenv("PINECONE_FRAME_INDEX", ""),
    "audio_index_name": os.getenv("PINECONE_AUDIO_INDEX", ""),
    "text_index_name": os.getenv("PINECONE_TRANSCRIPT_INDEX", ""),
    "desc_index_name": os.getenv("PINECONE_DESCRIPTION_INDEX", "")
}

# Processing configuration
BATCH_SIZE = 16  # Number of scenes to process in each batch
