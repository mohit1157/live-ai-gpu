# =============================================================================
# Avatar Rendering Service - GPU-accelerated Dockerfile
#
# Base: NVIDIA CUDA 12.4 + cuDNN 9 runtime on Ubuntu 22.04
# Target: AWS g5.xlarge (NVIDIA A10G, 24GB VRAM)
#
# Build stages:
# 1. Base: CUDA runtime + Python 3.11 + system deps
# 2. Dependencies: pip install (cached layer)
# 3. App: Copy application code
#
# Includes:
# - FFmpeg for video encoding (fallback when NVENC SDK not available)
# - TensorRT runtime (for optimized model inference)
# - CUDA toolkit (for custom CUDA kernels in 3DGS rasterizer)
# =============================================================================

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    # FFmpeg for video encoding (stub fallback for NVENC)
    ffmpeg \
    # Build tools for compiling CUDA extensions
    build-essential \
    cmake \
    ninja-build \
    git \
    # Networking tools for WebRTC
    libssl-dev \
    libffi-dev \
    # Font for stub renderer overlay text
    fonts-dejavu-core \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# Dependencies stage (cached)
# ---------------------------------------------------------------------------
FROM base AS deps

WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

# TODO: Uncomment when GPU dependencies are ready:
# RUN python -m pip install --no-cache-dir \
#     torch==2.2.0+cu124 \
#     torchvision==0.17.0+cu124 \
#     --index-url https://download.pytorch.org/whl/cu124
#
# RUN python -m pip install --no-cache-dir \
#     diff-gaussian-rasterization \
#     simple-knn \
#     tensorrt>=10.0.0 \
#     pynvml>=11.5.0

# ---------------------------------------------------------------------------
# Application stage
# ---------------------------------------------------------------------------
FROM deps AS app

WORKDIR /app

# Copy application code
COPY . .

# Create directory for model cache
RUN mkdir -p /tmp/liveai_models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8001/health'); r.raise_for_status()" || exit 1

EXPOSE 8001

# Run with uvicorn
# --workers 1: Single worker since we're GPU-bound (one GPU per container)
# --timeout-keep-alive 120: Keep WebSocket connections alive
# --limit-concurrency 50: Limit concurrent connections to avoid OOM
CMD ["python", "-m", "uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--timeout-keep-alive", "120", \
     "--limit-concurrency", "50", \
     "--log-level", "info"]
