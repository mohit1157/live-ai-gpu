# LiveAI Avatar Platform - Cloud Deployment Guide

## Architecture Overview

```
[Vercel]          [Railway]              [RunPod GPU Pod]
 Next.js    <-->   FastAPI API    <-->    Avatar    :8001
 Frontend          PostgreSQL             Voice     :8002
                   Redis                  Expression:8003
                   Celery Worker          Streaming :8004
```

## Prerequisites

### Accounts Required

| Service | Purpose | Signup |
|---------|---------|--------|
| Vercel | Frontend hosting (free tier) | https://vercel.com |
| Railway | Backend + DB + Redis | https://railway.app |
| RunPod | GPU compute for AI models | https://runpod.io |
| GitHub | Source code repository | https://github.com |

### CLI Tools

```bash
# Node.js (v18+)
# https://nodejs.org/

# Vercel CLI
npm install -g vercel
vercel login

# Railway CLI
npm install -g @railway/cli
railway login

# RunPod CLI (optional, for automated deployment)
# https://github.com/runpod/runpodctl
```

## Quick Start (Automated)

```bash
chmod +x deploy/deploy-all.sh
./deploy/deploy-all.sh
```

This will guide you through deploying all three services.

## Step-by-Step Deployment

### 1. Vercel (Frontend)

```bash
cd apps/web

# First time: link to Vercel project
vercel

# Deploy to production
vercel --prod
```

**Set environment variables** in the Vercel dashboard (Settings > Environment Variables):

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | `https://liveai-api.up.railway.app` |
| `NEXT_PUBLIC_WS_URL` | `wss://liveai-api.up.railway.app` |
| `NEXTAUTH_SECRET` | Generate with `openssl rand -base64 32` |
| `NEXTAUTH_URL` | `https://your-app.vercel.app` |

### 2. Railway (Backend)

```bash
# From project root
railway init --name liveai

# Add databases
railway add --plugin postgresql
railway add --plugin redis

# Deploy
railway up
```

**Set environment variables** in Railway dashboard:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (auto-linked) |
| `REDIS_URL` | `${{Redis.REDIS_URL}}` (auto-linked) |
| `SECRET_KEY` | Generate with `openssl rand -hex 32` |
| `CORS_ORIGINS` | `https://your-app.vercel.app` |
| `AVATAR_SERVICE_URL` | `https://<pod-id>-8001.proxy.runpod.net` |
| `VOICE_SERVICE_URL` | `https://<pod-id>-8002.proxy.runpod.net` |
| `EXPRESSION_SERVICE_URL` | `https://<pod-id>-8003.proxy.runpod.net` |
| `STREAMING_SERVICE_URL` | `https://<pod-id>-8004.proxy.runpod.net` |

Railway provides two deployment methods:

- **Nixpacks** (default): Auto-detects Python, uses `nixpacks.toml` and `railway.json`
- **Docker**: Uses `railway.toml` with `docker/api.Dockerfile`

### 3. RunPod (GPU Services)

#### Option A: Automated

```bash
export RUNPOD_API_KEY=your_api_key
./deploy/runpod/deploy.sh
```

#### Option B: Manual via Dashboard

1. Go to https://www.runpod.io/console/pods
2. Create a new pod:
   - **GPU**: NVIDIA A10G or RTX A4000 (24GB VRAM)
   - **Template**: RunPod PyTorch 2.x (CUDA 12.x)
   - **Container Disk**: 20GB
   - **Volume**: 50GB at `/workspace`
   - **Expose HTTP Ports**: `8001, 8002, 8003, 8004`
3. Open pod terminal:

```bash
cd /workspace
git clone <your-repo-url> live-ai
cd live-ai
chmod +x deploy/runpod/setup.sh
./deploy/runpod/setup.sh
```

#### Option C: Docker Image

```bash
# Build the GPU image
docker build -f deploy/runpod/Dockerfile.gpu -t liveai-gpu .

# Push to Docker Hub
docker tag liveai-gpu youruser/liveai-gpu:latest
docker push youruser/liveai-gpu:latest

# Use as RunPod custom template
```

## Connecting Everything

After all three services are deployed:

1. **Get RunPod URLs** from setup script output (or RunPod dashboard)
2. **Set GPU URLs** in Railway environment variables
3. **Get Railway URL** from Railway dashboard
4. **Set API URL** in Vercel environment variables
5. **Redeploy** both Vercel and Railway to pick up new env vars

## Testing the Deployment

```bash
# Frontend
curl -I https://your-app.vercel.app

# Backend health check
curl https://liveai-api.up.railway.app/health

# GPU services health check
curl https://<pod-id>-8001.proxy.runpod.net/health
curl https://<pod-id>-8002.proxy.runpod.net/health
curl https://<pod-id>-8003.proxy.runpod.net/health
curl https://<pod-id>-8004.proxy.runpod.net/health
```

## Cost Breakdown

### Vercel (Frontend)

| Plan | Cost | Includes |
|------|------|----------|
| Hobby (Free) | $0/mo | 100GB bandwidth, serverless functions |
| Pro | $20/mo | More bandwidth, team features |

### Railway (Backend)

| Component | Estimated Cost |
|-----------|---------------|
| API Server | ~$5/mo (512MB RAM) |
| PostgreSQL | ~$5/mo (1GB) |
| Redis | ~$3/mo (256MB) |
| **Total** | **~$10-15/mo** |

Railway offers $5 free credit per month on the Starter plan.

### RunPod (GPU)

| GPU | $/hr | VRAM | Best For |
|-----|------|------|----------|
| RTX A4000 | $0.31 | 16GB | Testing, light inference |
| A10G | $0.39 | 24GB | Production inference |
| RTX A5000 | $0.49 | 24GB | Good balance |
| A100 40GB | $1.64 | 40GB | Training + inference |

**Cost optimization tips**:
- Use Community Cloud for cheaper rates (30-50% less)
- Stop the pod when not in use (you only pay for active time)
- Use volume storage to persist models across restarts
- For always-on: consider RunPod Serverless instead

### Estimated Total Monthly Cost

| Scenario | Cost |
|----------|------|
| Development (occasional GPU) | ~$15-25/mo |
| Light production (GPU 8hr/day) | ~$85-120/mo |
| Full production (GPU 24/7) | ~$240-370/mo |

## File Reference

```
deploy/
  deploy-all.sh              # Master deployment script
  README.md                  # This file
  runpod/
    deploy.sh                # Automated RunPod pod creation
    setup.sh                 # RunPod in-pod setup script
    Dockerfile.gpu           # GPU services Docker image
    supervisord.conf         # Process manager for 4 services
    requirements-gpu.txt     # Combined Python requirements
    healthcheck.sh           # Health check script

apps/web/
  vercel.json                # Vercel deployment config
  .env.production            # Production env template

apps/api/
  railway.toml               # Railway Docker deployment config
  railway.json               # Railway Nixpacks deployment config
  nixpacks.toml              # Nixpacks build config
  Procfile                   # Process definitions for Railway
```

## Troubleshooting

### Vercel build fails
- Check `apps/web/package.json` has a valid `build` script
- Ensure all dependencies are in `dependencies` (not just `devDependencies`)
- Check build logs in Vercel dashboard

### Railway deploy fails
- Verify `pyproject.toml` has all dependencies listed
- Check that `alembic upgrade head` succeeds (DATABASE_URL must be set)
- View logs: `railway logs`

### RunPod services not starting
- Check GPU is available: `nvidia-smi`
- Check service logs: `tail -f /var/log/liveai/*_err.log`
- Verify supervisord: `supervisorctl status`
- Ensure ports are exposed in RunPod pod config

### Services cannot communicate
- Verify all URLs are correct (no trailing slashes)
- Check CORS settings in Railway backend
- Ensure RunPod ports are exposed as HTTP (not TCP)
- Test with curl from each service to verify connectivity
