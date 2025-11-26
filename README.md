# Video Embedding Pipeline - Backend

A terminal-only backend for processing videos and generating multimodal embeddings using ImageBind, scene detection, and Pinecone vector storage.

## Structure

```
.
├── src/                    # Main source code
│   ├── embedding/         # Embedding generation pipeline
│   └── retrieval/         # Retrieval module (placeholder)
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
└── README_SETUP.md        # Detailed setup instructions
```

## Quick Start

1. **Run setup:**
   ```bash
   ./setup.sh
   ```

2. **Configure Pinecone:**
   - Update `src/embedding/config.py` with your Pinecone API key
   - Create 4 indexes: `video-search`, `audio-search`, `text-search`, `desc-search`

3. **Run the pipeline:**
   ```bash
   python src/embedding/run_example.py
   ```

## Features

- 🎬 **Video Processing**: Downloads videos from pre-signed S3 URLs
- 📝 **Transcript Processing**: Downloads and processes transcripts from S3
- 🔍 **Scene Detection**: Automatic scene segmentation using PySceneDetect
- 🎨 **Multimodal Embeddings**: Generates embeddings for video, audio, text, and descriptions
- 📊 **Vector Storage**: Uploads embeddings to Pinecone for search

## Documentation

- **Setup Guide**: See `README_SETUP.md`
- **How to Run**: See `src/embedding/HOW_TO_RUN.md`
- **Configuration**: See `src/PLACEHOLDERS.md`

## Requirements

- Python 3.10+
- ffmpeg (system-wide)
- Pinecone account with 4 indexes
- Pre-signed S3 URLs for video and transcript
