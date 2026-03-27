"""Streaming Service - Manages live RTMP streaming sessions."""

import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

app = FastAPI(
    title="Streaming Service",
    description="Live RTMP streaming orchestration service",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# In-memory session store (stub)
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    avatar_id: str
    platform: str = Field(description="Streaming platform (e.g. youtube, twitch, custom)")
    stream_key: str
    rtmp_url: str
    avatar_service_url: str = "http://avatar-service:8001"


class SessionResponse(BaseModel):
    session_id: str
    status: str


class StreamHealthResponse(BaseModel):
    session_id: str
    status: str
    bitrate: int = Field(description="Current bitrate in kbps")
    fps: float
    frame_drops: int
    uptime: float = Field(description="Uptime in seconds")


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """Create a new streaming session."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "avatar_id": request.avatar_id,
        "platform": request.platform,
        "stream_key": request.stream_key,
        "rtmp_url": request.rtmp_url,
        "avatar_service_url": request.avatar_service_url,
        "status": "created",
        "started_at": None,
    }
    return SessionResponse(session_id=session_id, status="created")


@app.post("/sessions/{session_id}/start", response_model=SessionResponse)
async def start_session(session_id: str) -> SessionResponse:
    """Start an RTMP stream for the given session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id]["status"] = "live"
    _sessions[session_id]["started_at"] = time.time()
    return SessionResponse(session_id=session_id, status="live")


@app.post("/sessions/{session_id}/stop", response_model=SessionResponse)
async def stop_session(session_id: str) -> SessionResponse:
    """Stop an RTMP stream for the given session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id]["status"] = "stopped"
    return SessionResponse(session_id=session_id, status="stopped")


@app.get("/sessions/{session_id}/status", response_model=StreamHealthResponse)
async def session_status(session_id: str) -> StreamHealthResponse:
    """Get the health and status of a streaming session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    uptime = 0.0
    if session.get("started_at") and session["status"] == "live":
        uptime = time.time() - session["started_at"]

    return StreamHealthResponse(
        session_id=session_id,
        status=session["status"],
        bitrate=4500,
        fps=30.0,
        frame_drops=0,
        uptime=round(uptime, 2),
    )


@app.websocket("/sessions/{session_id}/feed")
async def session_feed(websocket: WebSocket, session_id: str) -> None:
    """Accept FLAME params, forward to avatar renderer, encode and push via RTMP."""
    if session_id not in _sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    frame_count = 0
    try:
        while True:
            data = await websocket.receive_text()
            frame_count += 1
            await websocket.send_json({
                "status": "ok",
                "frame": frame_count,
                "message": "Frame accepted and forwarded to renderer (stub)",
            })
    except WebSocketDisconnect:
        pass


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")
