"""
Configuration with hardcoded credentials (PLACEHOLDERS)
"""

# ============================================================================
# PLACEHOLDER: Pinecone Configuration
# ============================================================================
# TODO: Update these values with your Pinecone credentials
PINECONE_CONFIG = {
    "api_key": "pcsk_68fghf_5w37zk5oyZwVJdLx645ueUtfrLyGSb22yvdT2qyPRxBGG2BLrUvDhaK8shRPugW",  # PLACEHOLDER: Update with your Pinecone API key
    "video_index_name": "video-search",  # PLACEHOLDER: Update if different
    "audio_index_name": "audio-search",  # PLACEHOLDER: Update if different
    "text_index_name": "text-search",    # PLACEHOLDER: Update if different
    "desc_index_name": "desc-search"      # PLACEHOLDER: Update if different
}

# Processing configuration
BATCH_SIZE = 16  # Number of scenes to process in each batch

