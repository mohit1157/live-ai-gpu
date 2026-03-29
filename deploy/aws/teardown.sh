#!/usr/bin/env bash
# ===========================================================================
# LiveAI Platform - AWS GPU Teardown Script
#
# Stops services on the EC2 instance and optionally terminates it to save
# costs. Can also delete the entire CloudFormation stack.
#
# Usage:
#   ./teardown.sh                           # Stop containers only
#   ./teardown.sh --terminate               # Stop + terminate instance
#   ./teardown.sh --delete-stack            # Delete the entire CF stack
#   ./teardown.sh --region us-east-1        # Specify region
#   ./teardown.sh --stack-name liveai-gpu   # Specify stack name
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
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
STACK_NAME="liveai-gpu"
KEY_NAME="${KEY_NAME:-liveai-gpu-key}"
TERMINATE=false
DELETE_STACK=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --terminate)    TERMINATE=true; shift ;;
    --delete-stack) DELETE_STACK=true; shift ;;
    --region)       REGION="$2"; shift 2 ;;
    --stack-name)   STACK_NAME="$2"; shift 2 ;;
    --key-name)     KEY_NAME="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--terminate] [--delete-stack] [--region REGION] [--stack-name NAME]"
      exit 0
      ;;
    *) error "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Get instance details from CloudFormation
# ---------------------------------------------------------------------------
info "Retrieving instance details from stack: ${STACK_NAME}"

INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
  --output text 2>/dev/null || echo "")

PUBLIC_IP=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' \
  --output text 2>/dev/null || echo "")

if [[ -z "${INSTANCE_ID}" ]]; then
  error "Could not find instance in stack ${STACK_NAME}."
  if ${DELETE_STACK}; then
    warn "Proceeding with stack deletion anyway..."
  else
    exit 1
  fi
else
  info "Instance: ${INSTANCE_ID} (${PUBLIC_IP})"
fi

# ---------------------------------------------------------------------------
# Stop containers on the instance
# ---------------------------------------------------------------------------
if [[ -n "${PUBLIC_IP}" ]]; then
  info "Stopping containers on ${PUBLIC_IP}..."

  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -i "${HOME}/.ssh/${KEY_NAME}.pem" \
    "ubuntu@${PUBLIC_IP}" \
    "docker stop liveai-avatar liveai-voice liveai-expression liveai-streaming 2>/dev/null; \
     docker rm liveai-avatar liveai-voice liveai-expression liveai-streaming 2>/dev/null; \
     echo 'Containers stopped and removed.'" \
    2>/dev/null || warn "Could not SSH to instance. It may already be stopped."

  success "Containers stopped"
fi

# ---------------------------------------------------------------------------
# Terminate instance (stop only, does not delete stack)
# ---------------------------------------------------------------------------
if ${TERMINATE} && [[ -n "${INSTANCE_ID}" ]]; then
  echo ""
  warn "This will STOP the EC2 instance. You will not be charged for compute"
  warn "while it is stopped, but EBS volumes will still incur charges."
  echo ""
  read -p "Stop instance ${INSTANCE_ID}? (y/N): " CONFIRM
  if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
    info "Stopping instance ${INSTANCE_ID}..."
    aws ec2 stop-instances \
      --instance-ids "${INSTANCE_ID}" \
      --region "${REGION}" \
      --output text >/dev/null
    success "Instance ${INSTANCE_ID} is stopping."
    info "To restart later: aws ec2 start-instances --instance-ids ${INSTANCE_ID} --region ${REGION}"
  else
    info "Skipped instance stop."
  fi
fi

# ---------------------------------------------------------------------------
# Delete entire CloudFormation stack
# ---------------------------------------------------------------------------
if ${DELETE_STACK}; then
  echo ""
  warn "This will DELETE the entire CloudFormation stack, including:"
  warn "  - EC2 instance and its EBS volumes"
  warn "  - Security groups"
  warn "  - IAM roles"
  warn "  - Elastic IP (if allocated)"
  echo ""
  read -p "Delete stack ${STACK_NAME}? This cannot be undone. (y/N): " CONFIRM
  if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
    info "Deleting stack ${STACK_NAME}..."
    aws cloudformation delete-stack \
      --stack-name "${STACK_NAME}" \
      --region "${REGION}"

    info "Waiting for stack deletion (this may take a few minutes)..."
    aws cloudformation wait stack-delete-complete \
      --stack-name "${STACK_NAME}" \
      --region "${REGION}" 2>/dev/null || true

    success "Stack ${STACK_NAME} deleted."
  else
    info "Skipped stack deletion."
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  Teardown Summary${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
if ${DELETE_STACK}; then
  echo -e "  Stack ${STACK_NAME}: ${RED}DELETED${NC}"
elif ${TERMINATE}; then
  echo -e "  Instance ${INSTANCE_ID}: ${YELLOW}STOPPED${NC}"
  echo -e "  Stack ${STACK_NAME}: still exists"
  echo ""
  echo -e "  To restart:  ${CYAN}aws ec2 start-instances --instance-ids ${INSTANCE_ID} --region ${REGION}${NC}"
  echo -e "  To delete:   ${CYAN}./teardown.sh --delete-stack --stack-name ${STACK_NAME} --region ${REGION}${NC}"
else
  echo -e "  Containers: ${GREEN}STOPPED${NC}"
  echo -e "  Instance ${INSTANCE_ID}: ${YELLOW}STILL RUNNING${NC} (charges apply)"
  echo ""
  echo -e "  To stop instance:  ${CYAN}./teardown.sh --terminate${NC}"
  echo -e "  To delete stack:   ${CYAN}./teardown.sh --delete-stack${NC}"
fi
echo ""
