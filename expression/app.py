"""
Expression Service - Audio/video to FLAME expression parameter extraction.

Uses the ExpressionEngine for converting audio waveforms and video frames
into FLAME-compatible expression parameter sequences suitable for driving
Gaussian avatar rendering.

Endpoints:
  POST /audio-to-expression   - Generate expressions from audio
  POST /video-to-expression   - Extract expressions from video
  POST /landmarks-to-flame    - Convert MediaPipe landmarks to FLAME
  GET  /health                - Service health check
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from models.expression_engine import ExpressionEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
engine: Optional[ExpressionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the expression engine on startup."""
    global engine

    logger.info("Expression service starting up...")

    device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    engine = ExpressionEngine(device=device)

    logger.info("Expression service ready (device=%s)", device)
    yield
    logger.info("Expression service shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Expression Service",
    description="Audio/video to FLAME expression parameter extraction service",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class FLAMEParamDict(BaseModel):
    expression: list[float] = Field(description="52 FLAME expression coefficients")
    jaw_pose: list[float] = Field(description="3-dim jaw pose")
    eye_gaze: list[float] = Field(description="6-dim eye gaze")
    head_pose: list[float] = Field(description="6-dim head pose")


class AudioToExpressionRequest(BaseModel):
    audio_url: str = Field(description="URL or local path to audio file")
    fps: int = 30


class VideoToExpressionRequest(BaseModel):
    video_url: str = Field(description="URL or local path to video file")
    fps: int = 30


class ExpressionResponse(BaseModel):
    frames: list[FLAMEParamDict]
    fps: int
    duration: float


class LandmarksToFLAMERequest(BaseModel):
    landmarks: list[list[float]] = Field(
        description="478 MediaPipe face landmarks, each [x, y, z]"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu_available: bool = False
    engine_loaded: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/audio-to-expression", response_model=ExpressionResponse)
async def audio_to_expression(request: AudioToExpressionRequest) -> ExpressionResponse:
    """
    Generate FLAME expression parameters from an audio file.

    The audio is analyzed for speech energy to drive lip sync, with
    natural-looking blinks and head movement added automatically.
    """
    # Resolve audio path (URL or local path)
    audio_path = _resolve_media_path(request.audio_url)

    try:
        frames_data = engine.audio_to_expression(audio_path, fps=request.fps)
    except Exception as e:
        logger.error("audio_to_expression failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Expression extraction failed: {e}")

    frames = [
        FLAMEParamDict(
            expression=f["expression"],
            jaw_pose=f["jaw_pose"],
            eye_gaze=f["eye_gaze"],
            head_pose=f["head_pose"],
        )
        for f in frames_data
    ]

    duration = len(frames) / request.fps if frames else 0.0

    return ExpressionResponse(
        frames=frames,
        fps=request.fps,
        duration=duration,
    )


@app.post("/audio-to-expression/upload", response_model=ExpressionResponse)
async def audio_to_expression_upload(
    file: UploadFile = File(...),
    fps: int = Form(30),
) -> ExpressionResponse:
    """
    Generate FLAME expression parameters from an uploaded audio file.

    Alternative to the URL-based endpoint for direct file uploads.
    """
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        frames_data = engine.audio_to_expression(tmp_path, fps=fps)
    except Exception as e:
        logger.error("audio_to_expression_upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    frames = [
        FLAMEParamDict(
            expression=f["expression"],
            jaw_pose=f["jaw_pose"],
            eye_gaze=f["eye_gaze"],
            head_pose=f["head_pose"],
        )
        for f in frames_data
    ]

    duration = len(frames) / fps if frames else 0.0

    return ExpressionResponse(frames=frames, fps=fps, duration=duration)


@app.post("/video-to-expression", response_model=ExpressionResponse)
async def video_to_expression(request: VideoToExpressionRequest) -> ExpressionResponse:
    """Extract FLAME expression parameters from a video file."""
    video_path = _resolve_media_path(request.video_url)

    try:
        frames_data = engine.video_to_expression(video_path, fps=request.fps)
    except Exception as e:
        logger.error("video_to_expression failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    frames = [
        FLAMEParamDict(
            expression=f["expression"],
            jaw_pose=f["jaw_pose"],
            eye_gaze=f["eye_gaze"],
            head_pose=f["head_pose"],
        )
        for f in frames_data
    ]

    duration = len(frames) / request.fps if frames else 0.0

    return ExpressionResponse(
        frames=frames,
        fps=request.fps,
        duration=duration,
    )


@app.post("/landmarks-to-flame", response_model=FLAMEParamDict)
async def landmarks_to_flame(request: LandmarksToFLAMERequest) -> FLAMEParamDict:
    """Convert 478 MediaPipe face landmarks to FLAME parameters."""
    if len(request.landmarks) != 478:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 478 landmarks, got {len(request.landmarks)}",
        )

    landmarks_array = np.array(request.landmarks, dtype=np.float32)

    try:
        result = engine.landmarks_to_flame(landmarks_array)
    except Exception as e:
        logger.error("landmarks_to_flame failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return FLAMEParamDict(
        expression=result["expression"],
        jaw_pose=result["jaw_pose"],
        eye_gaze=result["eye_gaze"],
        head_pose=result["head_pose"],
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
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
        engine_loaded=engine is not None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_media_path(url_or_path: str) -> str:
    """
    Resolve a media URL or local path to a local file path.

    For URLs, downloads the file to a temp location.
    For local paths, returns as-is.
    """
    if url_or_path.startswith(("http://", "https://", "s3://")):
        # In production, download from URL or S3
        # STUB: return the URL as path (will trigger fallback in engine)
        logger.info("Media URL received: %s (stub: using fallback)", url_or_path)
        return url_or_path
    return url_or_path
