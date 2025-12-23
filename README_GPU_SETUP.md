# GPU Setup Guide

## Problem: CUDA Not Available

If you see `CUDA Available: False` when running `create_embeddings.py`, PyTorch is likely installed as CPU-only.

## Solution: Install PyTorch with CUDA Support

### Quick Fix (Recommended)

Run the installation script:

```bash
./install_pytorch_cuda.sh
```

This will:
1. Check for NVIDIA GPU
2. Uninstall CPU-only PyTorch
3. Install PyTorch 2.4.0 with CUDA 12.4 support
4. Verify CUDA is working

### Manual Installation

If you prefer to install manually:

```bash
# Uninstall CPU-only PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify installation
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Verify GPU Detection

After installation, verify CUDA is working:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Troubleshooting

**If CUDA is still not available after installation:**

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```
   Should show your GPU. If not, install NVIDIA drivers.

2. **Check CUDA runtime compatibility:**
   - PyTorch CUDA 12.4 requires CUDA runtime 12.4 or compatible
   - Your driver supports CUDA 13.0, which is backward compatible
   - If issues persist, try CUDA 12.1:
     ```bash
     pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
         --index-url https://download.pytorch.org/whl/cu121
     ```

3. **Check Python environment:**
   - Ensure you're using the same Python environment where PyTorch is installed
   - If using conda, activate the environment first:
     ```bash
     conda activate your_env_name
     ```

4. **Verify PyTorch installation:**
   ```bash
   python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   ```
   Should show CUDA version (not `None`).

### Expected Output After Fix

When running `create_embeddings.py`, you should see:

```
CUDA Available: True
CUDA Version: 12.4
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 4080
GPU Memory: 16.00 GB

✅ GPU ACCELERATION ENABLED
  Embedding Model Device: cuda:0
  Caption Model Device: cuda:0
  GPU: NVIDIA GeForce RTX 4080
  GPU Memory: 16.00 GB
```

### Performance Impact

- **With GPU:** Processing time ~1-2 minutes per video
- **Without GPU (CPU-only):** Processing time ~10-30+ minutes per video

GPU acceleration provides **10-100x speedup** for embedding generation.

