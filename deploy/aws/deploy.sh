#!/usr/bin/env bash
# ===========================================================================
# LiveAI Platform - AWS GPU Deployment Script
#
# Builds Docker images for all four GPU services, pushes them to ECR, launches
# a g5.xlarge EC2 instance with the Deep Learning AMI, and starts the services.
#
# Prerequisites:
#   - AWS CLI v2 configured with appropriate credentials
#   - Docker running locally
#   - SSH key pair created in AWS (name passed via --key-name or KEY_NAME env)
#
# Usage:
#   ./deploy.sh [--key-name my-key] [--region us-east-1] [--stack-name liveai-gpu]
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
step()    { echo -e "\n${CYAN}==> $*${NC}"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
STACK_NAME="liveai-gpu"
KEY_NAME="${KEY_NAME:-liveai-gpu-key}"
SERVICES=("avatar" "voice" "expression" "streaming")
ECR_PREFIX="liveai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --key-name)  KEY_NAME="$2"; shift 2 ;;
    --region)    REGION="$2"; shift 2 ;;
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--key-name NAME] [--region REGION] [--stack-name NAME]"
      exit 0
      ;;
    *) error "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 1: Verify prerequisites
# ---------------------------------------------------------------------------
step "Step 1/7: Checking prerequisites"

if ! command -v aws &>/dev/null; then
  error "AWS CLI not found. Install from https://aws.amazon.com/cli/"
  exit 1
fi
success "AWS CLI found: $(aws --version 2>&1 | head -1)"

if ! aws sts get-caller-identity &>/dev/null; then
  error "AWS CLI not configured. Run 'aws configure' first."
  exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
success "AWS account: ${ACCOUNT_ID}"

if ! command -v docker &>/dev/null; then
  error "Docker not found. Install Docker Desktop or Docker Engine."
  exit 1
fi
success "Docker found: $(docker --version)"

ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
info "ECR registry: ${ECR_REGISTRY}"
info "Region: ${REGION}"
info "Stack name: ${STACK_NAME}"
info "SSH key: ${KEY_NAME}"

# ---------------------------------------------------------------------------
# Step 2: Build Docker images
# ---------------------------------------------------------------------------
step "Step 2/7: Building Docker images"

for svc in "${SERVICES[@]}"; do
  info "Building ${ECR_PREFIX}/${svc}..."
  docker build \
    -t "${ECR_PREFIX}/${svc}:latest" \
    -f "${REPO_ROOT}/services/${svc}/Dockerfile" \
    "${REPO_ROOT}/services/${svc}" \
    2>&1 | tail -5
  success "Built ${ECR_PREFIX}/${svc}:latest"
done

# ---------------------------------------------------------------------------
# Step 3: Create ECR repos and push images
# ---------------------------------------------------------------------------
step "Step 3/7: Pushing images to ECR"

# Login to ECR
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}" 2>/dev/null
success "Logged into ECR"

for svc in "${SERVICES[@]}"; do
  REPO_NAME="${ECR_PREFIX}/${svc}"
  FULL_URI="${ECR_REGISTRY}/${REPO_NAME}:latest"

  # Create repo if it doesn't exist
  if ! aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" &>/dev/null; then
    info "Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository \
      --repository-name "${REPO_NAME}" \
      --region "${REGION}" \
      --image-scanning-configuration scanOnPush=true \
      --output text >/dev/null
    success "Created ${REPO_NAME}"
  fi

  # Tag and push
  docker tag "${REPO_NAME}:latest" "${FULL_URI}"
  info "Pushing ${FULL_URI}..."
  docker push "${FULL_URI}" 2>&1 | tail -3
  success "Pushed ${REPO_NAME}"
done

# ---------------------------------------------------------------------------
# Step 4: Deploy CloudFormation stack
# ---------------------------------------------------------------------------
step "Step 4/7: Deploying CloudFormation stack"

CF_TEMPLATE="${SCRIPT_DIR}/cloudformation.yml"

if ! [ -f "${CF_TEMPLATE}" ]; then
  error "CloudFormation template not found at ${CF_TEMPLATE}"
  exit 1
fi

# Check if stack exists
STACK_STATUS=""
if aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${REGION}" &>/dev/null; then
  STACK_STATUS=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query 'Stacks[0].StackStatus' \
    --output text)
  info "Stack ${STACK_NAME} exists with status: ${STACK_STATUS}"
fi

CF_ACTION="create-stack"
CF_WAIT="stack-create-complete"
if [[ -n "${STACK_STATUS}" && "${STACK_STATUS}" != "DELETE_COMPLETE" ]]; then
  CF_ACTION="update-stack"
  CF_WAIT="stack-update-complete"
  info "Updating existing stack..."
fi

aws cloudformation ${CF_ACTION} \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --template-body "file://${CF_TEMPLATE}" \
  --capabilities CAPABILITY_IAM \
  --parameters \
    ParameterKey=KeyPairName,ParameterValue="${KEY_NAME}" \
    ParameterKey=ECRRegistry,ParameterValue="${ECR_REGISTRY}" \
  --output text 2>/dev/null || {
    if [[ "${CF_ACTION}" == "update-stack" ]]; then
      warn "No stack updates to apply (stack is already up to date)"
    else
      error "Failed to create CloudFormation stack"
      exit 1
    fi
  }

info "Waiting for stack to reach a stable state (this may take 3-5 minutes)..."
aws cloudformation wait "${CF_WAIT}" \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" 2>/dev/null || true

success "CloudFormation stack deployed"

# ---------------------------------------------------------------------------
# Step 5: Get instance details
# ---------------------------------------------------------------------------
step "Step 5/7: Retrieving instance details"

INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
  --output text)

PUBLIC_IP=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' \
  --output text)

if [[ -z "${INSTANCE_ID}" || -z "${PUBLIC_IP}" ]]; then
  error "Could not retrieve instance details from CloudFormation outputs."
  error "Check the stack in the AWS Console."
  exit 1
fi

success "Instance ID: ${INSTANCE_ID}"
success "Public IP:   ${PUBLIC_IP}"

# ---------------------------------------------------------------------------
# Step 6: Run setup on the instance
# ---------------------------------------------------------------------------
step "Step 6/7: Setting up services on the instance"

info "Waiting for SSH to become available..."
for i in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
    -i "${HOME}/.ssh/${KEY_NAME}.pem" "ubuntu@${PUBLIC_IP}" "echo ok" &>/dev/null; then
    break
  fi
  sleep 10
done

# Upload and run the setup script
scp -o StrictHostKeyChecking=no \
  -i "${HOME}/.ssh/${KEY_NAME}.pem" \
  "${SCRIPT_DIR}/setup-instance.sh" \
  "ubuntu@${PUBLIC_IP}:/tmp/setup-instance.sh"

ssh -o StrictHostKeyChecking=no \
  -i "${HOME}/.ssh/${KEY_NAME}.pem" \
  "ubuntu@${PUBLIC_IP}" \
  "ECR_REGISTRY=${ECR_REGISTRY} AWS_REGION=${REGION} bash /tmp/setup-instance.sh"

success "Instance setup complete"

# ---------------------------------------------------------------------------
# Step 7: Update .env and output summary
# ---------------------------------------------------------------------------
step "Step 7/7: Updating configuration"

ENV_FILE="${REPO_ROOT}/.env"

# Build service URLs
AVATAR_URL="http://${PUBLIC_IP}:8001"
VOICE_URL="http://${PUBLIC_IP}:8002"
EXPRESSION_URL="http://${PUBLIC_IP}:8003"
STREAMING_URL="http://${PUBLIC_IP}:8004"

# Update or create .env
if [[ -f "${ENV_FILE}" ]]; then
  # Update existing entries (or append)
  for pair in \
    "AVATAR_SERVICE_URL=${AVATAR_URL}" \
    "VOICE_SERVICE_URL=${VOICE_URL}" \
    "EXPRESSION_SERVICE_URL=${EXPRESSION_URL}" \
    "STREAMING_SERVICE_URL=${STREAMING_URL}" \
    "GPU_INSTANCE_IP=${PUBLIC_IP}" \
    "GPU_INSTANCE_ID=${INSTANCE_ID}"; do
    KEY="${pair%%=*}"
    VAL="${pair#*=}"
    if grep -q "^${KEY}=" "${ENV_FILE}" 2>/dev/null; then
      sed -i "s|^${KEY}=.*|${KEY}=${VAL}|" "${ENV_FILE}"
    else
      echo "${KEY}=${VAL}" >> "${ENV_FILE}"
    fi
  done
  success "Updated ${ENV_FILE}"
else
  cat > "${ENV_FILE}" <<EOF
# GPU Service URLs (auto-generated by deploy.sh)
AVATAR_SERVICE_URL=${AVATAR_URL}
VOICE_SERVICE_URL=${VOICE_URL}
EXPRESSION_SERVICE_URL=${EXPRESSION_URL}
STREAMING_SERVICE_URL=${STREAMING_URL}
GPU_INSTANCE_IP=${PUBLIC_IP}
GPU_INSTANCE_ID=${INSTANCE_ID}
EOF
  success "Created ${ENV_FILE}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  LiveAI GPU Deployment Complete                     ${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
echo -e "  Instance:    ${CYAN}${INSTANCE_ID}${NC}"
echo -e "  Public IP:   ${CYAN}${PUBLIC_IP}${NC}"
echo -e "  SSH:         ${CYAN}ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}${NC}"
echo ""
echo -e "  Service URLs:"
echo -e "    Avatar:     ${CYAN}${AVATAR_URL}${NC}"
echo -e "    Voice:      ${CYAN}${VOICE_URL}${NC}"
echo -e "    Expression: ${CYAN}${EXPRESSION_URL}${NC}"
echo -e "    Streaming:  ${CYAN}${STREAMING_URL}${NC}"
echo ""
echo -e "  Health checks:"
echo -e "    curl ${AVATAR_URL}/health"
echo -e "    curl ${VOICE_URL}/health"
echo -e "    curl ${EXPRESSION_URL}/health"
echo -e "    curl ${STREAMING_URL}/health"
echo ""
echo -e "  ${YELLOW}To save costs, tear down with:${NC}"
echo -e "    ${CYAN}./teardown.sh --stack-name ${STACK_NAME} --region ${REGION}${NC}"
echo ""
