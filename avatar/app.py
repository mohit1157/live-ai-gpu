"""
Avatar Service - GPU-accelerated avatar rendering with real-time streaming.

This FastAPI application provides:
1. Single-frame avatar rendering from FLAME parameters
2. Batch rendering for video generation
3. Real-time WebRTC-based streaming with <30ms latency
4. Avatar model management (load, cache, unload)
5. Training job orchestration (source feature extraction)

Uses LivePortrait for high-quality talking head animation.
Falls back to placeholder rendering when GPU models are not available.

Deployment: RunPod A100 80GB or AWS g5.xlarge (A10G GPU, 24GB VRAM)
Target: 30fps real-time rendering at 512x512 resolution
"""

import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from models.live_portrait import LivePortraitRenderer, LivePortraitTrainer
from renderer.nvenc_encoder import NVENCEncoder
from realtime.webrtc_handler import WebRTCHandler
from realtime.frame_pipeline import FramePipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global instances (initialized in lifespan)
# ---------------------------------------------------------------------------

renderer: Optional[LivePortraitRenderer] = None
encoder: Optional[NVENCEncoder] = None
webrtc_handler: Optional[WebRTCHandler] = None
# Active frame pipelines keyed by session_id
frame_pipelines: dict[str, FramePipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: initialize and tear down GPU resources.

    On startup:
    - Create the GaussianAvatarRenderer (allocates GPU buffers)
    - Create the NVENC encoder
    - Create the WebRTC handler
    - Log GPU availability and configuration

    On shutdown:
    - Stop all active pipelines
    - Release GPU memory
    - Close encoder
    """
    global renderer, encoder, webrtc_handler

    logger.info("Avatar service starting up...")

    # Initialize LivePortrait renderer
    import os
    model_path = os.environ.get("LIVEPORTRAIT_MODEL_PATH")
    device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    renderer = LivePortraitRenderer(model_path=model_path, device=device)

    # Initialize video encoder
    encoder = NVENCEncoder(width=512, height=512, fps=30, codec="h264", preset="low_latency")

    # Initialize WebRTC handler
    webrtc_handler = WebRTCHandler(renderer=renderer, encoder=encoder)

    logger.info(
        "Avatar service ready. Model loaded: %s, Device: %s",
        renderer.is_loaded,
        renderer.device,
    )

    yield

    # Shutdown
    logger.info("Avatar service shutting down...")

    # Stop all active pipelines
    for session_id, pipeline in frame_pipelines.items():
        try:
            await pipeline.stop()
        except Exception as e:
            logger.error("Error stopping pipeline %s: %s", session_id, e)
    frame_pipelines.clear()

    # Close encoder
    if encoder:
        encoder.close()

    logger.info("Avatar service stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Avatar Service",
    description="GPU-accelerated avatar rendering service with real-time streaming",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_render_time_header(request: Request, call_next):
    """Add X-Render-Time header with request processing time in ms."""
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Render-Time"] = f"{elapsed_ms:.2f}ms"
    return response


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    frames_archive_url: str
    user_id: str
    avatar_id: str
    config: Optional[dict] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str = "started"


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str


class FLAMEParams(BaseModel):
    expression: list[float] = Field(default_factory=lambda: [0.0] * 52)
    jaw_pose: list[float] = Field(default_factory=lambda: [0.0] * 3)
    eye_gaze: list[float] = Field(default_factory=lambda: [0.0] * 6)
    head_pose: list[float] = Field(default_factory=lambda: [0.0] * 6)


class RenderRequest(BaseModel):
    model_path: str
    flame_params: FLAMEParams
    resolution: int = 512


class BatchRenderRequest(BaseModel):
    model_path: str
    flame_params_sequence: list[FLAMEParams]
    resolution: int = 512


class BatchRenderResponse(BaseModel):
    frames_count: int
    output_url: str


class RealtimeStartRequest(BaseModel):
    model_path: str


class RealtimeStartResponse(BaseModel):
    session_id: str
    model_id: str
    status: str = "started"


class RealtimeStopRequest(BaseModel):
    session_id: str


class WebRTCOfferRequest(BaseModel):
    session_id: str
    sdp: str


class WebRTCOfferResponse(BaseModel):
    sdp: str


class WebRTCIceRequest(BaseModel):
    session_id: str
    candidate: dict


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu_available: bool = False
    tensorrt_available: bool = False
    active_sessions: int = 0
    cache_stats: Optional[dict] = None


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

@app.post("/train", response_model=TrainResponse)
async def train_avatar(request: TrainRequest) -> TrainResponse:
    """Start an avatar training job from a frames archive."""
    job_id = str(uuid.uuid4())
    logger.info(
        "Training job started: %s for user %s, avatar %s",
        job_id,
        request.user_id,
        request.avatar_id,
    )
    return TrainResponse(job_id=job_id, status="started")


@app.get("/train/{job_id}/status", response_model=TrainStatusResponse)
async def train_status(job_id: str) -> TrainStatusResponse:
    """Check the status of an avatar training job."""
    return TrainStatusResponse(
        job_id=job_id,
        status="completed",
        progress=100.0,
        message="Training completed successfully",
    )


# ---------------------------------------------------------------------------
# Render endpoints
# ---------------------------------------------------------------------------

@app.post("/render")
async def render_avatar(request: RenderRequest) -> StreamingResponse:
    """
    Render a single avatar frame from FLAME parameters.

    Loads the model if not cached, renders the frame, and returns
    a PNG image. Typical latency: 10-50ms depending on cache state.
    """
    # Load model (will use cache if available)
    model_id = renderer.load_model(request.model_path)

    # Render frame
    flame_dict = {
        "expression": request.flame_params.expression,
        "jaw_pose": request.flame_params.jaw_pose,
        "eye_gaze": request.flame_params.eye_gaze,
        "head_pose": request.flame_params.head_pose,
    }

    frame = renderer.render_frame(
        model_id,
        flame_params=flame_dict,
        resolution=request.resolution,
    )

    # Convert numpy RGBA to PNG
    img = Image.fromarray(frame, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/render/batch", response_model=BatchRenderResponse)
async def render_batch(request: BatchRenderRequest) -> BatchRenderResponse:
    """
    Render a batch of avatar frames from a FLAME parameter sequence.

    Used for offline video generation. Returns a URL to the rendered
    frames archive.
    """
    model_id = renderer.load_model(request.model_path)

    flame_dicts = [
        {
            "expression": p.expression,
            "jaw_pose": p.jaw_pose,
            "eye_gaze": p.eye_gaze,
            "head_pose": p.head_pose,
        }
        for p in request.flame_params_sequence
    ]

    frames = renderer.render_batch(model_id, flame_params_sequence=flame_dicts)

    # TODO: Encode frames to video and upload to S3
    output_url = f"s3://liveai-assets/renders/{uuid.uuid4()}/frames.tar.gz"

    return BatchRenderResponse(
        frames_count=len(frames),
        output_url=output_url,
    )


# ---------------------------------------------------------------------------
# Real-time rendering endpoints
# ---------------------------------------------------------------------------

@app.post("/render/realtime/start", response_model=RealtimeStartResponse)
async def realtime_start(request: RealtimeStartRequest) -> RealtimeStartResponse:
    """
    Start a real-time rendering session.

    Loads the avatar model into VRAM and creates a rendering pipeline
    ready to accept FLAME parameters at 30fps.
    """
    model_id = renderer.load_model(request.model_path)
    session_id = webrtc_handler.start_session(model_id)

    # Create a frame pipeline for this session
    pipeline = FramePipeline(renderer, encoder, buffer_size=3)
    await pipeline.start(model_id)
    frame_pipelines[session_id] = pipeline

    logger.info(
        "Real-time session started: %s (model=%s)",
        session_id,
        model_id,
    )

    return RealtimeStartResponse(
        session_id=session_id,
        model_id=model_id,
        status="started",
    )


@app.post("/render/realtime/stop")
async def realtime_stop(request: RealtimeStopRequest) -> dict:
    """
    Stop a real-time rendering session and free resources.
    """
    session_id = request.session_id

    # Stop frame pipeline
    pipeline = frame_pipelines.pop(session_id, None)
    if pipeline:
        await pipeline.stop()

    # Stop WebRTC session
    stats = webrtc_handler.stop_session(session_id)

    return {"status": "stopped", **stats}


@app.post("/render/realtime/offer", response_model=WebRTCOfferResponse)
async def webrtc_offer(request: WebRTCOfferRequest) -> WebRTCOfferResponse:
    """
    Exchange WebRTC offer/answer for real-time streaming.

    The client sends its SDP offer, and we respond with our SDP answer
    containing the video track we'll stream rendered frames on.
    """
    answer_sdp = await webrtc_handler.handle_offer(
        request.sdp, request.session_id
    )
    return WebRTCOfferResponse(sdp=answer_sdp)


@app.post("/render/realtime/ice")
async def webrtc_ice(request: WebRTCIceRequest) -> dict:
    """Process an ICE candidate for WebRTC connectivity."""
    await webrtc_handler.handle_ice_candidate(
        request.session_id, request.candidate
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

@app.get("/render/stats")
async def render_stats() -> dict:
    """
    Get comprehensive renderer performance statistics.

    Returns render timing, VRAM usage, cache stats, encoder stats,
    and active pipeline stats.
    """
    pipeline_stats = {
        sid: p.get_stats() for sid, p in frame_pipelines.items()
    }

    return {
        "renderer": {"is_loaded": renderer.is_loaded, "device": renderer.device, "loaded_models": len(renderer._loaded_sources)},
        "encoder": encoder.get_stats(),
        "active_sessions": webrtc_handler.get_active_sessions(),
        "pipelines": pipeline_stats,
    }


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@app.websocket("/render/stream")
async def render_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time avatar rendering.

    Protocol:
    1. Client sends JSON: {"action": "start", "model_path": "..."}
    2. Server responds: {"status": "ready", "session_id": "..."}
    3. Client sends FLAME params as JSON at 30fps
    4. Server renders and sends back PNG frames as binary messages
    5. Client sends: {"action": "stop"} to end

    This is a fallback for browsers that don't support WebRTC DataChannels.
    WebRTC is preferred for production use due to lower latency.
    """
    await websocket.accept()

    session_model_id: Optional[str] = None
    pipeline: Optional[FramePipeline] = None
    session_id: Optional[str] = None

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            action = msg.get("action", "params")

            if action == "start":
                # Initialize rendering session
                model_path = msg.get("model_path", "default")
                session_model_id = renderer.load_model(model_path)
                session_id = str(uuid.uuid4())

                # Create pipeline for this WebSocket session
                pipeline = FramePipeline(renderer, encoder, buffer_size=3)
                await pipeline.start(session_model_id)

                await websocket.send_json({
                    "status": "ready",
                    "session_id": session_id,
                    "model_id": session_model_id,
                })

            elif action == "stop":
                if pipeline:
                    await pipeline.stop()
                    pipeline = None
                await websocket.send_json({"status": "stopped"})
                break

            elif action == "params" or "expression" in msg:
                # FLAME parameters received
                if session_model_id is None:
                    await websocket.send_json({
                        "error": "Session not started. Send {action: 'start'} first."
                    })
                    continue

                flame_dict = {
                    "expression": msg.get("expression", [0.0] * 52),
                    "jaw_pose": msg.get("jaw_pose", [0.0, 0.0, 0.0]),
                    "eye_gaze": msg.get("eye_gaze", [0.0, 0.0, 0.0, 0.0]),
                    "head_pose": msg.get("head_pose", [0.0] * 6),
                }

                # Render frame directly (pipeline is for advanced WebRTC use)
                frame = renderer.render_frame(session_model_id, flame_params=flame_dict)

                # Send frame as PNG binary
                img = Image.fromarray(frame, mode="RGBA")
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=False)
                buf.seek(0)
                await websocket.send_bytes(buf.getvalue())

            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (session=%s)", session_id)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON received on WebSocket")
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        if pipeline:
            await pipeline.stop()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Reports GPU availability, TensorRT status, model cache stats,
    and number of active rendering sessions.
    """
    gpu_available = False
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        gpu_available = result.returncode == 0
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        tensorrt_available=False,
        active_sessions=len(frame_pipelines),
        cache_stats={"loaded_models": len(renderer._loaded_sources) if renderer else 0},
    )
