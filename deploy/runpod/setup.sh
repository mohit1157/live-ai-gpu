#!/bin/bash
set -e

# ============================================
# LiveAI - RunPod GPU Setup Script
# ============================================
# Run this INSIDE your RunPod pod after it starts.
#
# Recommended RunPod config:
#   - GPU: RTX A4000 (16GB) for budget, A10G (24GB) for production
#   - Template: RunPod PyTorch 2.x (CUDA 12.x)
#   - Container Disk: 20GB
#   - Volume: 50GB at /workspace (persists across restarts)
#   - Expose HTTP ports: 8001, 8002, 8003, 8004
#
# Usage:
#   1. Create a RunPod pod with the config above
#   2. Open terminal in the pod
#   3. git clone <your-repo> /workspace/live-ai && cd /workspace/live-ai
#   4. chmod +x deploy/runpod/setup.sh
#   5. ./deploy/runpod/setup.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BOLD}================================================${NC}"
echo -e "${BOLD}  LiveAI - RunPod GPU Services Setup${NC}"
echo -e "${BOLD}================================================${NC}"
echo ""

# ============================================
# Step 1: Check GPU availability
# ============================================
echo -e "${YELLOW}[1/6] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo ""
else
    echo -e "${RED}WARNING: nvidia-smi not found. GPU may not be available.${NC}"
    echo "  Make sure you're running this on a GPU pod."
    echo ""
fi

# ============================================
# Step 2: Install system dependencies
# ============================================
echo -e "${YELLOW}[2/6] Installing system dependencies...${NC}"

# Check if we need to install system packages
if ! command -v supervisord &> /dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        ffmpeg \
        supervisor \
        curl \
        wget \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        2>/dev/null || echo -e "${YELLOW}  Some system packages may need manual install${NC}"
    rm -rf /var/lib/apt/lists/*
    echo -e "  ${GREEN}System dependencies installed${NC}"
else
    echo -e "  ${GREEN}System dependencies already present${NC}"
fi

echo ""

# ============================================
# Step 3: Install Python dependencies
# ============================================
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"

pip install --upgrade pip -q

# Install from combined requirements file
if [ -f "$PROJECT_ROOT/deploy/runpod/requirements-gpu.txt" ]; then
    pip install -r "$PROJECT_ROOT/deploy/runpod/requirements-gpu.txt" -q
    echo -e "  ${GREEN}Installed from requirements-gpu.txt${NC}"
else
    # Fallback: install individual service requirements
    echo "  Installing individual service requirements..."
    for SERVICE in avatar voice expression streaming; do
        REQ="$PROJECT_ROOT/services/$SERVICE/requirements.txt"
        if [ -f "$REQ" ]; then
            pip install -r "$REQ" -q
            echo -e "  ${GREEN}Installed $SERVICE requirements${NC}"
        fi
    done
fi

# Install PyTorch with CUDA if not present
python3 -c "import torch; print(f'PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo -e "  ${YELLOW}PyTorch not found. For real model inference, install:${NC}"
    echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
}

echo ""

# ============================================
# Step 4: Download model weights (stubs)
# ============================================
echo -e "${YELLOW}[4/6] Setting up model directories...${NC}"

MODEL_DIR="/workspace/models"
mkdir -p "$MODEL_DIR/flame" \
         "$MODEL_DIR/mediapipe" \
         "$MODEL_DIR/xtts" \
         "$MODEL_DIR/wav2lip"

# FLAME model
if [ ! -f "$MODEL_DIR/flame/flame_model.pkl" ]; then
    echo -e "  ${YELLOW}FLAME model not found at $MODEL_DIR/flame/${NC}"
    echo "    Download from: https://flame.is.tue.mpg.de/"
    echo "    Place flame_model.pkl in $MODEL_DIR/flame/"
else
    echo -e "  ${GREEN}FLAME model found${NC}"
fi

# MediaPipe face mesh
if [ ! -f "$MODEL_DIR/mediapipe/face_landmarker.task" ]; then
    echo -e "  ${YELLOW}MediaPipe model not found. Downloading...${NC}"
    wget -q -O "$MODEL_DIR/mediapipe/face_landmarker.task" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
        2>/dev/null || echo "    Download manually from MediaPipe docs"
    [ -f "$MODEL_DIR/mediapipe/face_landmarker.task" ] && \
        echo -e "  ${GREEN}MediaPipe model downloaded${NC}"
fi

# XTTS v2 voice model
if [ ! -d "$MODEL_DIR/xtts/v2" ]; then
    echo -e "  ${YELLOW}XTTS v2 model not found at $MODEL_DIR/xtts/${NC}"
    echo "    Will auto-download on first use via Coqui TTS"
    echo "    Or manually: python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')\""
else
    echo -e "  ${GREEN}XTTS v2 model found${NC}"
fi

echo ""

# ============================================
# Step 5: Start services with supervisord
# ============================================
echo -e "${YELLOW}[5/6] Starting GPU services...${NC}"

# Create log directory
mkdir -p /var/log/liveai /var/log/supervisor

# Copy supervisord config
if [ -f "$PROJECT_ROOT/deploy/runpod/supervisord.conf" ]; then
    cp "$PROJECT_ROOT/deploy/runpod/supervisord.conf" /etc/supervisor/conf.d/liveai.conf
fi

# Set environment for model cache
export MODEL_CACHE_DIR="$MODEL_DIR"

# Stop existing supervisord if running
if [ -f /var/run/supervisord.pid ]; then
    kill "$(cat /var/run/supervisord.pid)" 2>/dev/null || true
    sleep 2
fi

# Start supervisord
supervisord -c /etc/supervisor/conf.d/liveai.conf &
SUPERVISOR_PID=$!

echo -e "  ${GREEN}Supervisord started (PID: $SUPERVISOR_PID)${NC}"
echo "  Waiting for services to initialize..."
sleep 8

echo ""

# ============================================
# Step 6: Health checks and output URLs
# ============================================
echo -e "${YELLOW}[6/6] Running health checks...${NC}"

SERVICES=("Avatar:8001" "Voice:8002" "Expression:8003" "Streaming:8004")
ALL_HEALTHY=true

for SERVICE_PORT in "${SERVICES[@]}"; do
    NAME="${SERVICE_PORT%%:*}"
    PORT="${SERVICE_PORT##*:}"

    # Retry health check up to 3 times
    HEALTHY=false
    for ATTEMPT in 1 2 3; do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}[OK]  $NAME (:$PORT)${NC}"
            HEALTHY=true
            break
        fi
        sleep 2
    done

    if [ "$HEALTHY" = false ]; then
        echo -e "  ${YELLOW}[WAIT] $NAME (:$PORT) - still starting up${NC}"
        ALL_HEALTHY=false
    fi
done

echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}All services are healthy!${NC}"
else
    echo -e "${YELLOW}Some services are still starting. Check logs:${NC}"
    echo "  supervisorctl status"
    echo "  tail -f /var/log/liveai/avatar_err.log"
fi

echo ""
echo -e "${BOLD}================================================${NC}"
echo -e "${GREEN}  GPU Services Running!${NC}"
echo -e "${BOLD}================================================${NC}"
echo ""

# Get RunPod pod ID for proxy URLs
POD_ID="${RUNPOD_POD_ID:-$(hostname)}"

echo -e "  ${BOLD}RunPod Proxy URLs:${NC}"
echo ""
echo "    AVATAR_SERVICE_URL=https://${POD_ID}-8001.proxy.runpod.net"
echo "    VOICE_SERVICE_URL=https://${POD_ID}-8002.proxy.runpod.net"
echo "    EXPRESSION_SERVICE_URL=https://${POD_ID}-8003.proxy.runpod.net"
echo "    STREAMING_SERVICE_URL=https://${POD_ID}-8004.proxy.runpod.net"
echo ""
echo -e "  ${BOLD}Set these in your Railway backend environment variables.${NC}"
echo ""
echo -e "  ${BOLD}Management:${NC}"
echo "    supervisorctl status          - View service status"
echo "    supervisorctl restart avatar   - Restart a service"
echo "    supervisorctl tail -f avatar   - Follow service logs"
echo "    tail -f /var/log/liveai/*.log  - View all logs"
echo ""
echo -e "  ${BOLD}Test endpoints:${NC}"
echo "    curl http://localhost:8001/health"
echo "    curl http://localhost:8002/health"
echo "    curl http://localhost:8003/health"
echo "    curl http://localhost:8004/health"
echo ""
