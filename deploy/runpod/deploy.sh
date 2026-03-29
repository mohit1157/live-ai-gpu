#!/bin/bash
set -e

# ============================================
# LiveAI - RunPod GPU Deployment Script
# ============================================
# Automates creating and setting up a RunPod pod
# for all 4 GPU services (avatar, voice, expression, streaming).
#
# Prerequisites:
#   - RunPod account with API key
#   - runpodctl CLI installed (or uses curl API fallback)
#   - RUNPOD_API_KEY environment variable set
#
# Usage:
#   export RUNPOD_API_KEY=your_api_key
#   ./deploy/runpod/deploy.sh
# ============================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default pod configuration
GPU_TYPE="${GPU_TYPE:-NVIDIA RTX A4000}"
GPU_COUNT="${GPU_COUNT:-1}"
CONTAINER_DISK="${CONTAINER_DISK:-20}"
VOLUME_DISK="${VOLUME_DISK:-50}"
VOLUME_MOUNT="${VOLUME_MOUNT:-/workspace}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
POD_NAME="${POD_NAME:-liveai-gpu}"
TEMPLATE_ID="${TEMPLATE_ID:-}"  # RunPod PyTorch 2.x template

echo -e "${BOLD}================================================${NC}"
echo -e "${BOLD}  LiveAI - RunPod GPU Deployment${NC}"
echo -e "${BOLD}================================================${NC}"
echo ""

# ============================================
# Step 1: Check prerequisites
# ============================================
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check API key
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}ERROR: RUNPOD_API_KEY environment variable not set.${NC}"
    echo ""
    echo "  Get your API key from: https://www.runpod.io/console/user/settings"
    echo ""
    echo "  Then run:"
    echo "    export RUNPOD_API_KEY=your_api_key_here"
    echo "    ./deploy/runpod/deploy.sh"
    exit 1
fi

# Check for runpodctl or curl
USE_CLI=false
if command -v runpodctl &> /dev/null; then
    USE_CLI=true
    echo -e "  ${GREEN}runpodctl CLI found${NC}"
elif command -v curl &> /dev/null; then
    echo -e "  ${YELLOW}runpodctl not found, using curl API fallback${NC}"
    echo ""
    echo "  To install runpodctl (recommended):"
    echo "    # Linux/macOS:"
    echo "    wget -qO- https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -O /usr/local/bin/runpodctl"
    echo "    chmod +x /usr/local/bin/runpodctl"
    echo "    runpodctl config --apiKey \$RUNPOD_API_KEY"
    echo ""
else
    echo -e "${RED}ERROR: Neither runpodctl nor curl found. Install one of them.${NC}"
    exit 1
fi

echo -e "  ${GREEN}API key set${NC}"
echo ""

# ============================================
# Step 2: Create pod via API
# ============================================
echo -e "${YELLOW}[2/6] Creating RunPod pod...${NC}"
echo "  GPU: $GPU_TYPE x$GPU_COUNT"
echo "  Container disk: ${CONTAINER_DISK}GB"
echo "  Volume: ${VOLUME_DISK}GB at $VOLUME_MOUNT"
echo "  Cloud: $CLOUD_TYPE"
echo ""

if [ "$USE_CLI" = true ]; then
    # Use runpodctl CLI
    POD_ID=$(runpodctl create pod \
        --name "$POD_NAME" \
        --gpuType "$GPU_TYPE" \
        --gpuCount "$GPU_COUNT" \
        --containerDiskSize "$CONTAINER_DISK" \
        --volumeSize "$VOLUME_DISK" \
        --volumePath "$VOLUME_MOUNT" \
        --ports "8001/http,8002/http,8003/http,8004/http,22/tcp" \
        --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
        --cloudType "$CLOUD_TYPE" \
        2>&1 | grep -oP '[a-z0-9]{24,}' | head -1)
else
    # Use GraphQL API via curl
    QUERY=$(cat <<'GRAPHQL'
mutation {
  podFindAndDeployOnDemand(
    input: {
      name: "PODNAME"
      imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
      gpuTypeId: "GPUTYPE"
      gpuCount: GPUCOUNT
      containerDiskInGb: CONTAINERDISK
      volumeInGb: VOLUMEDISK
      volumeMountPath: "VOLUMEMOUNT"
      cloudType: CLOUDTYPE
      ports: "8001/http,8002/http,8003/http,8004/http,22/tcp"
      startSsh: true
    }
  ) {
    id
    imageName
    gpuCount
    machineId
    machine {
      podHostId
    }
  }
}
GRAPHQL
)
    # Map friendly GPU names to RunPod GPU type IDs
    case "$GPU_TYPE" in
        *"A4000"*)  GPU_ID="NVIDIA RTX A4000" ;;
        *"A10G"*)   GPU_ID="NVIDIA A10G" ;;
        *"A5000"*)  GPU_ID="NVIDIA RTX A5000" ;;
        *"A100"*)   GPU_ID="NVIDIA A100 80GB PCIe" ;;
        *)          GPU_ID="$GPU_TYPE" ;;
    esac

    QUERY="${QUERY//PODNAME/$POD_NAME}"
    QUERY="${QUERY//GPUTYPE/$GPU_ID}"
    QUERY="${QUERY//GPUCOUNT/$GPU_COUNT}"
    QUERY="${QUERY//CONTAINERDISK/$CONTAINER_DISK}"
    QUERY="${QUERY//VOLUMEDISK/$VOLUME_DISK}"
    QUERY="${QUERY//VOLUMEMOUNT/$VOLUME_MOUNT}"
    QUERY="${QUERY//CLOUDTYPE/$CLOUD_TYPE}"

    RESPONSE=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$(echo "$QUERY" | tr '\n' ' ' | sed 's/"/\\"/g')\"}")

    POD_ID=$(echo "$RESPONSE" | grep -oP '"id"\s*:\s*"[^"]*"' | head -1 | grep -oP '"[^"]*"$' | tr -d '"')

    if [ -z "$POD_ID" ]; then
        echo -e "${RED}ERROR: Failed to create pod.${NC}"
        echo "API Response: $RESPONSE"
        echo ""
        echo "Common issues:"
        echo "  - No available GPUs of that type (try a different GPU or COMMUNITY cloud)"
        echo "  - Insufficient funds in your RunPod account"
        echo "  - Invalid API key"
        exit 1
    fi
fi

echo -e "  ${GREEN}Pod created: $POD_ID${NC}"
echo ""

# ============================================
# Step 3: Wait for pod to be ready
# ============================================
echo -e "${YELLOW}[3/6] Waiting for pod to be ready...${NC}"

MAX_WAIT=300  # 5 minutes
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ "$USE_CLI" = true ]; then
        STATUS=$(runpodctl get pod "$POD_ID" 2>&1 | grep -i "status" | head -1 || true)
    else
        STATUS_RESPONSE=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\": \"{ pod(input: { podId: \\\"$POD_ID\\\" }) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}")
        STATUS=$(echo "$STATUS_RESPONSE" | grep -oP '"desiredStatus"\s*:\s*"[^"]*"' | grep -oP '"[^"]*"$' | tr -d '"')
    fi

    if echo "$STATUS" | grep -qi "RUNNING"; then
        echo -e "  ${GREEN}Pod is running!${NC}"
        break
    fi

    echo -e "  Waiting... ($ELAPSED/${MAX_WAIT}s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo -e "${YELLOW}WARNING: Pod may still be starting. Check RunPod dashboard.${NC}"
fi

echo ""

# ============================================
# Step 4: Copy service code to the pod
# ============================================
echo -e "${YELLOW}[4/6] Copying service code to pod...${NC}"

if [ "$USE_CLI" = true ]; then
    # Use runpodctl to transfer files
    echo "  Syncing services directory..."
    runpodctl send "$PROJECT_ROOT/services" --podId "$POD_ID" --podPath "/workspace/live-ai/services/" 2>/dev/null || true
    runpodctl send "$PROJECT_ROOT/deploy/runpod" --podId "$POD_ID" --podPath "/workspace/live-ai/deploy/runpod/" 2>/dev/null || true
    echo -e "  ${GREEN}Files synced${NC}"
else
    echo -e "  ${YELLOW}Cannot auto-copy files without runpodctl.${NC}"
    echo ""
    echo "  Please copy files manually:"
    echo "    1. Open RunPod web terminal for pod $POD_ID"
    echo "    2. Run:"
    echo "       cd /workspace"
    echo "       git clone <your-repo-url> live-ai"
    echo ""
    echo "  Or install runpodctl for automatic file transfer."
fi

echo ""

# ============================================
# Step 5: Run setup script on the pod
# ============================================
echo -e "${YELLOW}[5/6] Running setup on pod...${NC}"

if [ "$USE_CLI" = true ]; then
    echo "  Executing setup script on pod..."
    runpodctl exec "$POD_ID" -- bash -c "cd /workspace/live-ai && chmod +x deploy/runpod/setup.sh && ./deploy/runpod/setup.sh" 2>&1 || {
        echo -e "  ${YELLOW}Auto-setup may not have completed. Run manually:${NC}"
        echo "    1. Open terminal in RunPod dashboard"
        echo "    2. cd /workspace/live-ai"
        echo "    3. ./deploy/runpod/setup.sh"
    }
else
    echo -e "  ${YELLOW}Cannot auto-execute without runpodctl. Run setup manually:${NC}"
    echo ""
    echo "    1. Open terminal in RunPod dashboard for pod: $POD_ID"
    echo "    2. cd /workspace/live-ai"
    echo "    3. chmod +x deploy/runpod/setup.sh"
    echo "    4. ./deploy/runpod/setup.sh"
fi

echo ""

# ============================================
# Step 6: Output proxy URLs
# ============================================
echo -e "${YELLOW}[6/6] Deployment complete!${NC}"
echo ""
echo -e "${BOLD}================================================${NC}"
echo -e "${GREEN}  RunPod GPU Services Deployed${NC}"
echo -e "${BOLD}================================================${NC}"
echo ""
echo -e "  ${BOLD}Pod ID:${NC} $POD_ID"
echo ""
echo -e "  ${BOLD}Service URLs:${NC}"
echo -e "  ${BLUE}Avatar:${NC}     https://${POD_ID}-8001.proxy.runpod.net"
echo -e "  ${BLUE}Voice:${NC}      https://${POD_ID}-8002.proxy.runpod.net"
echo -e "  ${BLUE}Expression:${NC} https://${POD_ID}-8003.proxy.runpod.net"
echo -e "  ${BLUE}Streaming:${NC}  https://${POD_ID}-8004.proxy.runpod.net"
echo ""
echo -e "  ${BOLD}Environment variables for your API backend:${NC}"
echo ""
echo "    AVATAR_SERVICE_URL=https://${POD_ID}-8001.proxy.runpod.net"
echo "    VOICE_SERVICE_URL=https://${POD_ID}-8002.proxy.runpod.net"
echo "    EXPRESSION_SERVICE_URL=https://${POD_ID}-8003.proxy.runpod.net"
echo "    STREAMING_SERVICE_URL=https://${POD_ID}-8004.proxy.runpod.net"
echo ""
echo -e "  ${BOLD}Management:${NC}"
echo "    Dashboard: https://www.runpod.io/console/pods"
echo "    Stop pod:  runpodctl stop pod $POD_ID"
echo "    Start pod: runpodctl start pod $POD_ID"
echo "    Remove:    runpodctl remove pod $POD_ID"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo "    1. Set the GPU service URLs in your Railway backend environment"
echo "    2. Verify health checks: curl https://${POD_ID}-8001.proxy.runpod.net/health"
echo "    3. Monitor logs via RunPod web terminal"
echo ""
