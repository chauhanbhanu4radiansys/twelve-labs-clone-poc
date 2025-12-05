# How to Run app-code-ref.py

## Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Step 2: Install Python Dependencies

```bash
cd /home/ai-team/Documents/bhanu_chauhan_rsys/video-search

# Install all Python packages
pip install -r requirements-app-ref.txt

# Install ImageBind from GitHub (required)
pip install git+https://github.com/facebookresearch/ImageBind.git
```

**Note:** If you encounter issues with ImageBind, you can also install it manually:
```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install -e .
cd ..
```

## Step 3: Run the Streamlit App

### Option A: Accessible Only on Local Machine (Default)
```bash
cd /home/ai-team/Documents/bhanu_chauhan_rsys/video-search

# Run the app (only accessible from localhost)
streamlit run app-code-ref.py
```

The app will open in your browser at `http://localhost:8501`

### Option B: Accessible Over Network (Recommended for Remote Access)
```bash
cd /home/ai-team/Documents/bhanu_chauhan_rsys/video-search

# Run the app accessible from network (use your server's IP)
streamlit run app-code-ref.py --server.address 0.0.0.0 --server.port 8501
```

Then access it from any machine on your network at:
- `http://192.168.0.64:8501` (from other machines)
- `http://localhost:8501` (from the same machine)

**Note:** Replace `192.168.0.64` with your actual server IP address if different.

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install dependencies:
```bash
pip install -r requirements-app-ref.txt
```

### Issue 2: "ffmpeg: command not found"
**Solution:** Install ffmpeg:
```bash
sudo apt-get install ffmpeg  # Ubuntu/Debian
```

### Issue 3: "ImageBind not found"
**Solution:** Install ImageBind:
```bash
pip install git+https://github.com/facebookresearch/ImageBind.git
```

### Issue 4: CUDA/GPU errors
**Solution:** If you don't have a GPU, PyTorch will fall back to CPU (slower but works).

### Issue 5: Connection errors to MongoDB/Pinecone/MinIO
**Solution:** Check your credentials in `app-code-ref.py` (lines 222-239) and ensure:
- MongoDB is accessible
- Pinecone API key is valid
- MinIO is running (if using local MinIO)

## Quick Test

To test if everything is installed correctly:
```bash
python3 -c "import streamlit; import torch; import cv2; import pinecone; import pymongo; print('All imports successful!')"
```

## Running in Background

To run the app in the background (accessible over network):
```bash
# Accessible over network
nohup streamlit run app-code-ref.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &

# Or only localhost
nohup streamlit run app-code-ref.py > streamlit.log 2>&1 &
```

To stop it:
```bash
pkill -f "streamlit run"
```

## Network Access Configuration

### Make Streamlit Accessible Over Network

By default, Streamlit only listens on `localhost` (127.0.0.1), making it inaccessible from other machines. To allow network access:

**Method 1: Command Line Flag (Temporary)**
```bash
streamlit run app-code-ref.py --server.address 0.0.0.0 --server.port 8501
```

**Method 2: Create Streamlit Config File (Permanent)**
Create a config file at `~/.streamlit/config.toml`:
```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false
EOF
```

Then run normally:
```bash
streamlit run app-code-ref.py
```

### Firewall Configuration

If you still can't access it, check your firewall:

**Ubuntu/Debian:**
```bash
# Allow port 8501 through firewall
sudo ufw allow 8501/tcp
sudo ufw reload
```

**Check if port is listening:**
```bash
sudo netstat -tlnp | grep 8501
# or
sudo ss -tlnp | grep 8501
```

### Security Warning

⚠️ **WARNING:** Making Streamlit accessible over the network exposes your application. For production:
- Use a reverse proxy (nginx) with SSL/TLS
- Implement authentication
- Use a VPN or restrict access to specific IPs
- Consider using Streamlit Cloud or other hosted solutions

