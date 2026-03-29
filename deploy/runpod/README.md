# RunPod GPU Deployment

## Quick Start

### 1. Create a RunPod Pod

Go to [runpod.io](https://runpod.io) and create a pod:

- **GPU**: RTX A4000 ($0.31/hr) for testing, A100 40GB ($1.64/hr) for production quality
- **Template**: RunPod PyTorch 2.x (CUDA 12.x)
- **Container Disk**: 20GB
- **Volume Disk**: 50GB (persists model weights across restarts)
- **Expose HTTP Ports**: 8001, 8002, 8003, 8004

### 2. Connect & Setup

```bash
# Open terminal in RunPod web UI, then:
cd /workspace
git clone <your-repo-url> live-ai
cd live-ai
chmod +x deploy/runpod/setup.sh
./deploy/runpod/setup.sh
```

### 3. Update Pi Configuration

Copy the RunPod proxy URLs from the setup output to your Pi's `.env`:

```bash
# On your Pi:
nano deploy/pi/.env

# Update these lines:
AVATAR_SERVICE_URL=https://<pod-id>-8001.proxy.runpod.net
VOICE_SERVICE_URL=https://<pod-id>-8002.proxy.runpod.net
EXPRESSION_SERVICE_URL=https://<pod-id>-8003.proxy.runpod.net
STREAMING_SERVICE_URL=https://<pod-id>-8004.proxy.runpod.net

# Restart Pi services:
cd deploy/pi
docker compose -f docker-compose.pi.yml restart api celery-worker
```

## Cost Estimates

| GPU | $/hr | Best For |
|-----|------|----------|
| RTX A4000 (16GB) | $0.31 | Testing, light inference |
| RTX A5000 (24GB) | $0.49 | Good balance |
| A100 40GB | $1.64 | Training + high quality inference |
| A100 80GB | $2.21 | Large models, batch processing |

Tip: Use "Community Cloud" for cheaper rates. Stop the pod when not in use.
