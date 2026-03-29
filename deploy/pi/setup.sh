#!/bin/bash
set -e

# ============================================
# LiveAI - Raspberry Pi 4 Setup Script
# ============================================
# Run this on your Pi after cloning the repo:
#   chmod +x deploy/pi/setup.sh
#   ./deploy/pi/setup.sh

echo "================================================"
echo "  LiveAI - Raspberry Pi 4 Setup"
echo "================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check we're on ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${YELLOW}Warning: Expected ARM64 (aarch64), got $ARCH${NC}"
    echo "This script is optimized for Raspberry Pi 4/5"
fi

# Get Pi IP
PI_IP=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}Detected Pi IP: $PI_IP${NC}"

# ============================================
# Step 1: Install Docker if not present
# ============================================
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}Docker installed. You may need to log out and back in.${NC}"
fi

if ! command -v docker compose &> /dev/null; then
    echo -e "${YELLOW}Installing Docker Compose plugin...${NC}"
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

echo -e "${GREEN}Docker version:${NC}"
docker --version

# ============================================
# Step 2: Configure swap (important for 8GB Pi)
# ============================================
echo -e "${YELLOW}Configuring swap to 4GB...${NC}"
sudo dphys-swapfile swapoff 2>/dev/null || true
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile 2>/dev/null || true
sudo dphys-swapfile setup 2>/dev/null || true
sudo dphys-swapfile swapon 2>/dev/null || true
echo -e "${GREEN}Swap configured${NC}"

# ============================================
# Step 3: Create .env file
# ============================================
ENV_FILE="deploy/pi/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp deploy/pi/.env.example "$ENV_FILE"

    # Set the Pi IP
    sed -i "s/PI_IP=.*/PI_IP=$PI_IP/" "$ENV_FILE"

    # Generate JWT secret
    JWT_SECRET=$(openssl rand -hex 32)
    sed -i "s/JWT_SECRET=.*/JWT_SECRET=$JWT_SECRET/" "$ENV_FILE"

    echo -e "${GREEN}.env created at $ENV_FILE${NC}"
    echo -e "${YELLOW}IMPORTANT: Edit $ENV_FILE to add your CLAUDE_API_KEY${NC}"
else
    echo -e "${GREEN}.env already exists${NC}"
fi

# ============================================
# Step 4: Build and start services
# ============================================
echo ""
echo -e "${YELLOW}Building Docker images (this takes 10-20 min on Pi 4)...${NC}"
echo "Go grab a coffee..."
echo ""

cd deploy/pi
docker compose -f docker-compose.pi.yml build

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""

# ============================================
# Step 5: Start services
# ============================================
echo -e "${YELLOW}Starting services...${NC}"
docker compose -f docker-compose.pi.yml up -d

# Wait for postgres
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Run migrations
echo -e "${YELLOW}Running database migrations...${NC}"
docker compose -f docker-compose.pi.yml exec api alembic upgrade head

echo ""
echo "================================================"
echo -e "${GREEN}  LiveAI is running!${NC}"
echo "================================================"
echo ""
echo "  Web UI:      http://$PI_IP:3000"
echo "  API:         http://$PI_IP:8000"
echo "  API Docs:    http://$PI_IP:8000/docs"
echo "  MinIO:       http://$PI_IP:9001  (minioadmin/minioadmin)"
echo ""
echo "  Access from any device on your network!"
echo ""
echo "================================================"
echo "  Next Steps:"
echo "================================================"
echo ""
echo "  1. Edit deploy/pi/.env and add your CLAUDE_API_KEY"
echo "  2. Deploy GPU services to RunPod (see deploy/runpod/)"
echo "  3. Update .env with RunPod service URLs"
echo "  4. Restart: docker compose -f docker-compose.pi.yml restart"
echo ""
echo "  Useful commands:"
echo "    View logs:  docker compose -f docker-compose.pi.yml logs -f"
echo "    Stop:       docker compose -f docker-compose.pi.yml down"
echo "    Restart:    docker compose -f docker-compose.pi.yml restart"
echo ""
