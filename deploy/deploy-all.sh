#!/bin/bash
set -e

# ============================================
# LiveAI - Master Deployment Script
# ============================================
# Deploys the entire LiveAI Avatar Platform:
#   1. Frontend  -> Vercel (Next.js)
#   2. Backend   -> Railway (FastAPI + PostgreSQL + Redis)
#   3. GPU       -> RunPod (Avatar, Voice, Expression, Streaming)
#
# Usage:
#   chmod +x deploy/deploy-all.sh
#   ./deploy/deploy-all.sh
#
# Options:
#   ./deploy/deploy-all.sh --frontend-only
#   ./deploy/deploy-all.sh --backend-only
#   ./deploy/deploy-all.sh --gpu-only

# ============================================
# Colors and formatting
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
DEPLOY_FRONTEND=true
DEPLOY_BACKEND=true
DEPLOY_GPU=true

case "${1:-}" in
    --frontend-only) DEPLOY_BACKEND=false; DEPLOY_GPU=false ;;
    --backend-only)  DEPLOY_FRONTEND=false; DEPLOY_GPU=false ;;
    --gpu-only)      DEPLOY_FRONTEND=false; DEPLOY_BACKEND=false ;;
esac

# ============================================
# Helper functions
# ============================================
print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}   $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}   $1${NC}"
}

print_error() {
    echo -e "${RED}   $1${NC}"
}

print_info() {
    echo -e "${BLUE}   $1${NC}"
}

check_command() {
    local cmd=$1
    local install_msg=$2
    if command -v "$cmd" &> /dev/null; then
        print_success "$cmd found: $(command -v "$cmd")"
        return 0
    else
        print_error "$cmd not found"
        echo -e "     ${YELLOW}Install: $install_msg${NC}"
        return 1
    fi
}

# ============================================
# Banner
# ============================================
echo ""
echo -e "${BOLD}${MAGENTA}"
echo "  _     _          _    ___ "
echo " | |   (_)_   ____| |  / _ \\ ___ "
echo " | |   | \\ \\ / / _ \\ | / /_\\/  __| "
echo " | |___| |\\ V /  __/ |/ /   \\ (_ |"
echo " |_____|_| \\_/ \\___|_|\\/    \\___| "
echo ""
echo -e "  ${CYAN}Avatar Platform - Cloud Deployment${NC}"
echo -e "${NC}"

# ============================================
# Step 0: Check prerequisites
# ============================================
print_header "Checking Prerequisites"

MISSING_DEPS=0

print_step "Checking required tools..."

check_command "git" "https://git-scm.com/downloads" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_command "node" "https://nodejs.org/ (v18+)" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_command "npm" "Comes with Node.js" || MISSING_DEPS=$((MISSING_DEPS + 1))

echo ""
print_step "Checking deployment CLIs..."

VERCEL_OK=true
RAILWAY_OK=true

if [ "$DEPLOY_FRONTEND" = true ]; then
    check_command "vercel" "npm install -g vercel" || { VERCEL_OK=false; MISSING_DEPS=$((MISSING_DEPS + 1)); }
fi

if [ "$DEPLOY_BACKEND" = true ]; then
    check_command "railway" "npm install -g @railway/cli  (then: railway login)" || { RAILWAY_OK=false; MISSING_DEPS=$((MISSING_DEPS + 1)); }
fi

echo ""

if [ $MISSING_DEPS -gt 0 ]; then
    echo -e "${YELLOW}Some tools are missing. Install them and re-run this script.${NC}"
    echo ""
    echo "Quick install commands:"
    echo ""
    if [ "$VERCEL_OK" = false ]; then
        echo "  # Vercel CLI"
        echo "  npm install -g vercel"
        echo "  vercel login"
        echo ""
    fi
    if [ "$RAILWAY_OK" = false ]; then
        echo "  # Railway CLI"
        echo "  npm install -g @railway/cli"
        echo "  railway login"
        echo ""
    fi
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "Aborting."
        exit 1
    fi
fi

# ============================================
# Step 1: Initialize git repo if needed
# ============================================
print_header "Git Repository"

cd "$PROJECT_ROOT"

if [ -d .git ]; then
    print_success "Git repo already initialized"
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    print_info "Current branch: $BRANCH"
else
    print_step "Initializing git repository..."
    git init
    git add -A
    git commit -m "Initial commit - LiveAI Avatar Platform"
    print_success "Git repo initialized with initial commit"
fi

echo ""

# ============================================
# Step 2: Deploy Frontend to Vercel
# ============================================
if [ "$DEPLOY_FRONTEND" = true ]; then
    print_header "Deploying Frontend to Vercel"

    if [ "$VERCEL_OK" = false ]; then
        print_warn "Skipping Vercel deployment (CLI not installed)"
        print_info "Install vercel CLI and run: cd apps/web && vercel --prod"
    else
        cd "$PROJECT_ROOT/apps/web"

        print_step "Deploying to Vercel..."
        echo ""

        # Check if already linked to Vercel project
        if [ -d .vercel ]; then
            print_info "Vercel project already linked"
        else
            print_info "First time setup - Vercel will ask you to link/create a project"
        fi

        # Deploy to production
        VERCEL_URL=$(vercel --prod --yes 2>&1 | tail -1) || {
            print_warn "Vercel deployment needs manual setup."
            echo ""
            echo "  Run these commands manually:"
            echo "    cd $PROJECT_ROOT/apps/web"
            echo "    vercel              # First time: link project"
            echo "    vercel --prod       # Deploy to production"
            echo ""
            echo "  Then set environment variables in Vercel dashboard:"
            echo "    NEXT_PUBLIC_API_URL = https://liveai-api.up.railway.app"
            echo "    NEXT_PUBLIC_WS_URL  = wss://liveai-api.up.railway.app"
            echo "    NEXTAUTH_SECRET     = (generate with: openssl rand -base64 32)"
            echo "    NEXTAUTH_URL        = https://your-app.vercel.app"
            VERCEL_URL="(pending)"
        }

        if [ "$VERCEL_URL" != "(pending)" ]; then
            print_success "Frontend deployed to: $VERCEL_URL"
        fi

        cd "$PROJECT_ROOT"
    fi

    echo ""
fi

# ============================================
# Step 3: Deploy Backend to Railway
# ============================================
if [ "$DEPLOY_BACKEND" = true ]; then
    print_header "Deploying Backend to Railway"

    if [ "$RAILWAY_OK" = false ]; then
        print_warn "Skipping Railway deployment (CLI not installed)"
        print_info "Install railway CLI and run the commands below manually"
    else
        cd "$PROJECT_ROOT"

        print_step "Setting up Railway project..."

        # Check if Railway project is linked
        if [ -f .railway/config.json ] || railway status &>/dev/null; then
            print_info "Railway project already linked"
        else
            print_info "Creating new Railway project..."
            railway init --name liveai 2>/dev/null || {
                print_warn "Railway project setup needs manual intervention."
                echo ""
                echo "  Run: railway init --name liveai"
            }
        fi

        # Deploy API service
        print_step "Deploying API service..."
        railway up --service api --detach 2>/dev/null || {
            print_warn "Railway deployment needs manual setup."
            echo ""
            echo "  Steps to deploy manually:"
            echo ""
            echo "  1. Create project:  railway init --name liveai"
            echo "  2. Add PostgreSQL:  railway add --plugin postgresql"
            echo "  3. Add Redis:       railway add --plugin redis"
            echo "  4. Deploy API:      railway up"
            echo ""
        }

        # Output Railway info
        RAILWAY_URL=$(railway domain 2>/dev/null || echo "(pending)")
        if [ "$RAILWAY_URL" != "(pending)" ]; then
            print_success "Backend deployed to: https://$RAILWAY_URL"
        fi

        cd "$PROJECT_ROOT"
    fi

    echo ""

    # Railway environment variables guide
    print_step "Railway environment variables to set:"
    echo ""
    echo "  # In Railway dashboard (https://railway.app/dashboard):"
    echo ""
    echo "  DATABASE_URL=\${{Postgres.DATABASE_URL}}"
    echo "  REDIS_URL=\${{Redis.REDIS_URL}}"
    echo "  SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo 'generate-a-secret-key')"
    echo "  CORS_ORIGINS=https://liveai.vercel.app"
    echo "  AVATAR_SERVICE_URL=https://<runpod-pod-id>-8001.proxy.runpod.net"
    echo "  VOICE_SERVICE_URL=https://<runpod-pod-id>-8002.proxy.runpod.net"
    echo "  EXPRESSION_SERVICE_URL=https://<runpod-pod-id>-8003.proxy.runpod.net"
    echo "  STREAMING_SERVICE_URL=https://<runpod-pod-id>-8004.proxy.runpod.net"
    echo ""
fi

# ============================================
# Step 4: Deploy GPU Services to RunPod
# ============================================
if [ "$DEPLOY_GPU" = true ]; then
    print_header "GPU Services (RunPod)"

    print_warn "RunPod deployment requires manual steps (GPU provisioning)."
    echo ""
    echo -e "  ${BOLD}Option A: Automated (requires RUNPOD_API_KEY)${NC}"
    echo ""
    echo "    export RUNPOD_API_KEY=your_api_key"
    echo "    ./deploy/runpod/deploy.sh"
    echo ""
    echo -e "  ${BOLD}Option B: Manual via RunPod Dashboard${NC}"
    echo ""
    echo "    1. Go to https://www.runpod.io/console/pods"
    echo "    2. Create a new pod:"
    echo "       - GPU: NVIDIA A10G or RTX A4000 (24GB VRAM)"
    echo "       - Template: RunPod PyTorch 2.x (CUDA 12.x)"
    echo "       - Container Disk: 20GB"
    echo "       - Volume: 50GB at /workspace"
    echo "       - Expose HTTP Ports: 8001, 8002, 8003, 8004"
    echo "    3. Open pod terminal and run:"
    echo "       cd /workspace"
    echo "       git clone <your-repo-url> live-ai"
    echo "       cd live-ai"
    echo "       chmod +x deploy/runpod/setup.sh"
    echo "       ./deploy/runpod/setup.sh"
    echo ""
    echo -e "  ${BOLD}Option C: Docker (pre-built image)${NC}"
    echo ""
    echo "    docker build -f deploy/runpod/Dockerfile.gpu -t liveai-gpu ."
    echo "    # Push to Docker Hub, then use as RunPod custom template"
    echo ""
fi

# ============================================
# Summary
# ============================================
print_header "Deployment Summary"

echo -e "  ${BOLD}Service        Target      Status${NC}"
echo -e "  ─────────────────────────────────────────"

if [ "$DEPLOY_FRONTEND" = true ]; then
    FURL="${VERCEL_URL:-(not deployed)}"
    echo -e "  Frontend       Vercel      ${GREEN}$FURL${NC}"
fi

if [ "$DEPLOY_BACKEND" = true ]; then
    BURL="${RAILWAY_URL:-(not deployed)}"
    echo -e "  Backend        Railway     ${GREEN}$BURL${NC}"
fi

if [ "$DEPLOY_GPU" = true ]; then
    echo -e "  GPU Services   RunPod      ${YELLOW}(manual setup required)${NC}"
fi

echo ""

print_header "Next Steps"

echo "  1. Set environment variables in each platform's dashboard"
echo "  2. Connect the services:"
echo "     - Vercel: Set NEXT_PUBLIC_API_URL to Railway backend URL"
echo "     - Railway: Set GPU service URLs from RunPod"
echo "     - Railway: Add PostgreSQL and Redis plugins"
echo "  3. Test the deployment:"
echo "     - Frontend: curl https://your-app.vercel.app"
echo "     - Backend:  curl https://your-api.up.railway.app/health"
echo "     - GPU:      curl https://<pod-id>-8001.proxy.runpod.net/health"
echo ""
echo -e "  ${BOLD}Estimated monthly costs (light usage):${NC}"
echo "     Vercel:   Free (hobby tier)"
echo "     Railway:  ~\$5-10/mo (starter plan)"
echo "     RunPod:   ~\$0.31-0.49/hr (on-demand, stop when not in use)"
echo ""
echo -e "${GREEN}Deployment script complete!${NC}"
echo ""
