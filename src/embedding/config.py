"""
Configuration with hardcoded credentials (PLACEHOLDERS)
"""

# ============================================================================
# PLACEHOLDER: Pinecone Configuration
# ============================================================================
# TODO: Update these values with your Pinecone credentials
PINECONE_CONFIG = {
    "api_key": "YOUR_PINECONE_API_KEY",  # PLACEHOLDER: Update with your Pinecone API key
    "video_index_name": "video-search",  # PLACEHOLDER: Update if different
    "audio_index_name": "audio-search",  # PLACEHOLDER: Update if different
    "text_index_name": "text-search",    # PLACEHOLDER: Update if different
    "desc_index_name": "desc-search"      # PLACEHOLDER: Update if different
}

# Processing configuration
BATCH_SIZE = 16  # Number of scenes to process in each batch

