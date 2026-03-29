#!/usr/bin/env bash
# ===========================================================================
# LiveAI Platform - EC2 Instance Setup Script
#
# Runs ON the GPU EC2 instance after launch. Installs Docker, NVIDIA
# Container Toolkit, pulls images from ECR, and starts all 4 GPU services.
#
# Expected environment variables:
#   ECR_REGISTRY  - e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com
#   AWS_REGION    - e.g. us-east-1
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

ECR_REGISTRY="${ECR_REGISTRY:?ECR_REGISTRY environment variable is required}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# ---------------------------------------------------------------------------
# Step 1: Install Docker if needed
# ---------------------------------------------------------------------------
info "Step 1: Checking Docker installation..."

if ! command -v docker &>/dev/null; then
  info "Installing Docker..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker.io
  sudo systemctl enable docker
  sudo systemctl start docker
  sudo usermod -aG docker ubuntu
  success "Docker installed"
else
  success "Docker already installed: $(docker --version)"
fi

# ---------------------------------------------------------------------------
# Step 2: Install NVIDIA Container Toolkit if needed
# ---------------------------------------------------------------------------
info "Step 2: Checking NVIDIA Container Toolkit..."

if ! dpkg -l | grep -q nvidia-container-toolkit; then
  info "Installing NVIDIA Container Toolkit..."

  # Add NVIDIA GPG key and repo
  distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit

  # Configure Docker to use NVIDIA runtime
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  success "NVIDIA Container Toolkit installed"
else
  success "NVIDIA Container Toolkit already installed"
fi

# Verify GPU access
if nvidia-smi &>/dev/null; then
  success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  warn "nvidia-smi not available. GPU drivers may not be installed."
  warn "The Deep Learning AMI should have drivers pre-installed."
fi

# ---------------------------------------------------------------------------
# Step 3: Login to ECR
# ---------------------------------------------------------------------------
info "Step 3: Logging into ECR..."

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}" 2>/dev/null
success "Logged into ECR: ${ECR_REGISTRY}"

# ---------------------------------------------------------------------------
# Step 4: Pull all service images
# ---------------------------------------------------------------------------
info "Step 4: Pulling service images..."

SERVICES=("avatar" "voice" "expression" "streaming")
PORTS=(8001 8002 8003 8004)

for svc in "${SERVICES[@]}"; do
  IMAGE="${ECR_REGISTRY}/liveai/${svc}:latest"
  info "Pulling ${IMAGE}..."
  docker pull "${IMAGE}" 2>&1 | tail -2
  success "Pulled liveai/${svc}"
done

# ---------------------------------------------------------------------------
# Step 5: Stop existing containers (if any)
# ---------------------------------------------------------------------------
info "Step 5: Stopping existing containers..."

for svc in "${SERVICES[@]}"; do
  CONTAINER_NAME="liveai-${svc}"
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    info "Removed old container: ${CONTAINER_NAME}"
  fi
done

# ---------------------------------------------------------------------------
# Step 6: Start containers with GPU access
# ---------------------------------------------------------------------------
info "Step 6: Starting service containers..."

# Create a shared network for inter-service communication
docker network create liveai-net 2>/dev/null || true

for i in "${!SERVICES[@]}"; do
  svc="${SERVICES[$i]}"
  port="${PORTS[$i]}"
  CONTAINER_NAME="liveai-${svc}"
  IMAGE="${ECR_REGISTRY}/liveai/${svc}:latest"

  info "Starting ${CONTAINER_NAME} on port ${port}..."
  docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --network liveai-net \
    -p "${port}:${port}" \
    --restart unless-stopped \
    -e "SERVICE_PORT=${port}" \
    -e "NVIDIA_VISIBLE_DEVICES=all" \
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" \
    --shm-size=2g \
    "${IMAGE}"
  success "Started ${CONTAINER_NAME} on port ${port}"
done

# ---------------------------------------------------------------------------
# Step 7: Wait for health checks
# ---------------------------------------------------------------------------
info "Step 7: Waiting for health checks..."

MAX_WAIT=120
INTERVAL=5
ELAPSED=0

all_healthy=false

while [[ ${ELAPSED} -lt ${MAX_WAIT} ]]; do
  all_healthy=true
  for i in "${!SERVICES[@]}"; do
    svc="${SERVICES[$i]}"
    port="${PORTS[$i]}"
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      : # healthy
    else
      all_healthy=false
    fi
  done

  if ${all_healthy}; then
    break
  fi

  sleep ${INTERVAL}
  ELAPSED=$((ELAPSED + INTERVAL))
  info "Waiting for services... (${ELAPSED}s / ${MAX_WAIT}s)"
done

echo ""
echo "======================================================"
echo "  Service Status"
echo "======================================================"

for i in "${!SERVICES[@]}"; do
  svc="${SERVICES[$i]}"
  port="${PORTS[$i]}"
  if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo -e "  liveai-${svc}  :${port}  ${GREEN}HEALTHY${NC}"
  else
    echo -e "  liveai-${svc}  :${port}  ${RED}UNHEALTHY${NC}"
  fi
done

echo "======================================================"

if ${all_healthy}; then
  success "All services are healthy and running."
else
  warn "Some services may not be healthy yet. Check logs with:"
  echo "  docker logs liveai-<service-name>"
fi
