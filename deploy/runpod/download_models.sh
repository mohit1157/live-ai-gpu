#!/bin/bash
# Download all AI models for LiveAI GPU services
# Run this once after pod creation. Models are cached in /workspace/models/

set -e

echo "=== LiveAI Model Downloader ==="

# Fish Speech S2 (~3GB)
echo "[1/4] Downloading Fish Speech S2..."
if [ ! -f /workspace/models/fish-speech/config.json ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/fish-speech-1.5', local_dir='/workspace/models/fish-speech')
print('Fish Speech downloaded')
" || echo "Fish Speech download failed - will retry on first use"
else
    echo "Fish Speech already cached"
fi

# LivePortrait (~2GB)
echo "[2/4] Downloading LivePortrait models..."
if [ ! -d /workspace/models/liveportrait/base_models ]; then
    cd /workspace/LivePortrait
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('KwaiVGI/LivePortrait', local_dir='/workspace/models/liveportrait')
print('LivePortrait downloaded')
" || echo "LivePortrait download failed"
else
    echo "LivePortrait already cached"
fi

# RMBG v2.0 (~200MB)
echo "[3/4] Downloading RMBG v2.0..."
if [ ! -f /workspace/models/rmbg/config.json ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('briaai/RMBG-2.0', local_dir='/workspace/models/rmbg')
print('RMBG downloaded')
" || echo "RMBG download failed"
else
    echo "RMBG already cached"
fi

# MuseTalk (~1.5GB)
echo "[4/4] Downloading MuseTalk models..."
if [ ! -d /workspace/models/musetalk ]; then
    cd /workspace/MuseTalk
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='/workspace/models/musetalk')
print('MuseTalk downloaded')
" || echo "MuseTalk download failed"
else
    echo "MuseTalk already cached"
fi

echo ""
echo "=== Model download complete ==="
echo "Total disk usage:"
du -sh /workspace/models/*/
