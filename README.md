# Video Prism: AI-Powered Video Search and Analysis

A comprehensive video processing and search application using multimodal embeddings (ImageBind), scene detection, transcription, and vector search.

## Features

- 🎬 **Video Upload & Processing**: Upload videos and automatically process them with scene detection
- 🔍 **Multimodal Search**: Search videos using text, images, or audio queries
- 📊 **AI Analysis**: Get AI-powered analysis of video content using OpenAI
- 🌐 **Multilingual Support**: Supports Indian languages with automatic translation
- 🎯 **Scene Detection**: Automatic scene detection and segmentation
- 📝 **Transcription**: Automatic audio transcription with Whisper and IndicConformer

## Prerequisites

- Python 3.10+
- ffmpeg (for video/audio processing)
- CUDA-capable GPU (recommended) or CPU
- Access to:
  - MinIO/S3 (object storage)
  - MongoDB (metadata database)
  - Pinecone (vector database)
  - OpenAI API (optional, for translation/analysis)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chauhanbhanu4radiansys/twelve-labs-clone-poc.git
    cd twelve-labs-clone-poc
    ```

2.  **Create and Activate a Conda Environment (Recommended):**
    It is highly recommended to use Python 3.10 or 3.11.
    ```bash
    conda create --name videoprism python=3.10 -y
    conda activate videoprism
    ```

3.  **Run the Installation Script:**
    This script will install all dependencies in the correct order to prevent build errors.
    ```bash
    bash install_dependencies.sh
    ```

4.  **Install System Dependencies:**
    ```bash
    # Ubuntu/Debian
    sudo apt-get install ffmpeg
    
    # macOS
    brew install ffmpeg
    ```

5.  **Set up Configuration:**
    ```bash
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    # Edit .streamlit/secrets.toml with your credentials
    ```

6.  **Set up Pinecone Indexes:**
    Create 4 indexes in your Pinecone account with:
    *   **Dimension:** `1024`
    *   **Metric:** `cosine`
    *   **Names:** `video-search`, `audio-search`, `text-search`, `desc-search`
    
For detailed troubleshooting, see the comments in the `install_dependencies.sh` script.

## Usage

Run the Streamlit application:

```