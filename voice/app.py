"""
Voice Service - GPU-accelerated voice cloning and TTS synthesis.

Uses XTTS v2 for voice cloning and text-to-speech. Falls back to silent
audio stubs when the GPU model is not available.

Endpoints:
  POST /clone           - Start a voice cloning job
  GET  /clone/{id}/status - Check cloning job progress
  POST /synthesize      - Generate speech from text (returns WAV)
  WS   /synthesize/stream - WebSocket streaming TTS
  GET  /health          - Service health check
"""

import asyncio
import io
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.xtts_v2 import XTTSVoiceCloner

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
voice_cloner: Optional[XTTSVoiceCloner] = None
clone_jobs: dict[str, dict] = {}  # job_id -> {status, progress, message, output_dir}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the voice cloner on startup."""
    global voice_cloner

    logger.info("Voice service starting up...")

    model_path = os.environ.get("XTTS_MODEL_PATH")
    device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"

    voice_cloner = XTTSVoiceCloner(model_path=model_path, device=device)
    logger.info("Voice service ready (device=%s, model_loaded=%s)", device, voice_cloner.is_loaded)

    yield

    logger.info("Voice service shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice Service",
    description="GPU-accelerated voice cloning and TTS service using XTTS v2",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class CloneRequest(BaseModel):
    audio_samples_urls: list[str] = Field(description="URLs to audio samples for voice cloning")
    user_id: str
    avatar_id: str


class CloneResponse(BaseModel):
    job_id: str
    status: str = "started"


class CloneStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    voice_model_path: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(description="Text to synthesize into speech")
    voice_model_path: str = Field(description="Path to the cloned voice model")
    language: str = "en"


class SynthesizeStreamRequest(BaseModel):
    text: str
    voice_model_path: str
    language: str = "en"


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu_available: bool = False
    model_loaded: bool = False
    active_clone_jobs: int = 0


# ---------------------------------------------------------------------------
# Clone endpoints
# ---------------------------------------------------------------------------
@app.post("/clone", response_model=CloneResponse)
async def clone_voice(request: CloneRequest) -> CloneResponse:
    """
    Start a voice cloning job from audio samples.

    The job runs asynchronously. Poll /clone/{job_id}/status for progress.
    """
    job_id = str(uuid.uuid4())
    output_dir = os.path.join(
        tempfile.gettempdir(), "liveai_voice", request.user_id, request.avatar_id
    )

    clone_jobs[job_id] = {
        "status": "started",
        "progress": 0.0,
        "message": "Job queued",
        "output_dir": output_dir,
        "voice_model_path": None,
    }

    # Run cloning in background
    asyncio.get_event_loop().run_in_executor(
        None, _run_clone_job, job_id, request.audio_samples_urls, output_dir
    )

    logger.info(
        "Clone job started: %s (user=%s, avatar=%s, %d samples)",
        job_id, request.user_id, request.avatar_id, len(request.audio_samples_urls),
    )
    return CloneResponse(job_id=job_id, status="started")


def _run_clone_job(job_id: str, audio_paths: list[str], output_dir: str) -> None:
    """Background worker for voice cloning."""
    try:
        def progress_cb(pct: float, msg: str) -> None:
            clone_jobs[job_id]["progress"] = pct
            clone_jobs[job_id]["message"] = msg
            clone_jobs[job_id]["status"] = "running"

        result_dir = voice_cloner.clone_voice(
            audio_paths=audio_paths,
            output_dir=output_dir,
            progress_callback=progress_cb,
        )

        clone_jobs[job_id]["status"] = "completed"
        clone_jobs[job_id]["progress"] = 100.0
        clone_jobs[job_id]["message"] = "Voice cloning completed successfully"
        clone_jobs[job_id]["voice_model_path"] = result_dir

    except Exception as e:
        logger.error("Clone job %s failed: %s", job_id, e, exc_info=True)
        clone_jobs[job_id]["status"] = "failed"
        clone_jobs[job_id]["message"] = str(e)


@app.get("/clone/{job_id}/status", response_model=CloneStatusResponse)
async def clone_status(job_id: str) -> CloneStatusResponse:
    """Check the status of a voice cloning job."""
    job = clone_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return CloneStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        voice_model_path=job.get("voice_model_path"),
    )


# ---------------------------------------------------------------------------
# Synthesis endpoints
# ---------------------------------------------------------------------------
@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> StreamingResponse:
    """
    Synthesize speech from text using a cloned voice model.

    Returns a WAV audio file.
    """
    try:
        audio = voice_cloner.synthesize(
            text=request.text,
            voice_model_path=request.voice_model_path,
            language=request.language,
        )

        # Convert numpy array to WAV bytes
        wav_bytes = _numpy_to_wav_bytes(audio)

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except Exception as e:
        logger.error("Synthesis failed: %s", e, exc_info=True)
        # Fallback: generate silent audio with estimated duration
        duration = max(1.0, len(request.text.split()) / 150 * 60)
        wav_bytes = XTTSVoiceCloner.generate_silent_wav_bytes(duration)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )


@app.websocket("/synthesize/stream")
async def synthesize_stream(websocket: WebSocket) -> None:
    """
    WebSocket streaming TTS.

    Protocol:
    1. Client sends JSON: {"text": "...", "voice_model_path": "...", "language": "en"}
    2. Server streams WAV audio chunks as binary messages
    3. Server sends JSON: {"status": "done"} when complete
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            voice_model_path = data.get("voice_model_path", "")
            language = data.get("language", "en")

            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue

            # Stream audio chunks
            for chunk in voice_cloner.synthesize_streaming(
                text=text,
                voice_model_path=voice_model_path,
                language=language,
            ):
                wav_chunk = _numpy_to_wav_bytes(chunk)
                await websocket.send_bytes(wav_chunk)

            await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Reports GPU availability, model status, and active job count.
    """
    gpu_available = False
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        gpu_available = result.returncode == 0
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        model_loaded=voice_cloner.is_loaded if voice_cloner else False,
        active_clone_jobs=sum(
            1 for j in clone_jobs.values() if j["status"] == "running"
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _numpy_to_wav_bytes(audio: "np.ndarray") -> bytes:
    """Convert a float32 numpy array to WAV bytes."""
    import struct
    import numpy as np

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    sample_rate = 22050
    num_channels = 1
    bits_per_sample = 16
    data_size = len(audio_int16) * num_channels * (bits_per_sample // 8)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * (bits_per_sample // 8)))
    buf.write(struct.pack("<H", num_channels * (bits_per_sample // 8)))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())

    return buf.getvalue()
