"""
Double-buffered frame pipeline for hiding rendering latency.

The key insight: rendering, encoding, and network transmission can all
happen in parallel for different frames. While frame N is being sent
to the client, frame N+1 is being encoded, and frame N+2 is being
rendered. This pipeline hides all three latencies behind each other.

Architecture:
    [Params Queue] -> [Render Worker] -> [Encode Worker] -> [Output Queue]
                         (GPU thread)      (GPU/CPU thread)   (Network thread)

Without pipelining: total_latency = render + encode + send = ~20ms
With pipelining: total_latency = max(render, encode, send) = ~10ms
Effective throughput: 100+ fps (bottlenecked by render at ~8ms/frame)

The pipeline uses:
- asyncio queues for producer/consumer coordination
- Threading for GPU operations (GIL released during CUDA calls)
- Pre-allocated buffers to avoid allocation in the hot path

Current status: STUB using asyncio queues with simulated timing.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..renderer.gaussian_renderer import GaussianAvatarRenderer
from ..renderer.nvenc_encoder import NVENCEncoder

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Rolling statistics for pipeline stages."""
    render_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    encode_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    total_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    frames_submitted: int = 0
    frames_completed: int = 0
    frames_dropped: int = 0

    def avg_render_ms(self) -> float:
        return sum(self.render_times_ms) / len(self.render_times_ms) if self.render_times_ms else 0

    def avg_encode_ms(self) -> float:
        return sum(self.encode_times_ms) / len(self.encode_times_ms) if self.encode_times_ms else 0

    def avg_total_ms(self) -> float:
        return sum(self.total_times_ms) / len(self.total_times_ms) if self.total_times_ms else 0


class FramePipeline:
    """
    Double-buffered frame pipeline for hiding latency.

    While frame N is being sent to the client:
    - Frame N+1 is being encoded
    - Frame N+2 is being rendered

    This hides encoder and network latency behind render time.
    Uses asyncio + threading for GPU/CPU overlap.

    Usage:
        pipeline = FramePipeline(renderer, encoder, buffer_size=3)
        await pipeline.start(model_id)

        # In the input handler (30fps):
        await pipeline.submit_params(flame_params)

        # In the output handler:
        encoded = await pipeline.get_encoded_frame()
        if encoded:
            send_to_client(encoded)

        await pipeline.stop()
    """

    def __init__(
        self,
        renderer: GaussianAvatarRenderer,
        encoder: NVENCEncoder,
        buffer_size: int = 3,
    ) -> None:
        """
        Initialize the frame pipeline.

        Args:
            renderer: The Gaussian splatting renderer.
            encoder: The NVENC video encoder.
            buffer_size: Number of frames to buffer at each stage.
                Higher values = smoother output but more latency.
                For real-time: 2-3. For video export: 8-16.
        """
        self._renderer = renderer
        self._encoder = encoder
        self._buffer_size = buffer_size

        # Pipeline queues
        self._params_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue(
            maxsize=buffer_size
        )
        self._render_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(
            maxsize=buffer_size
        )
        self._output_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(
            maxsize=buffer_size
        )

        # Pipeline state
        self._model_id: Optional[str] = None
        self._is_running: bool = False
        self._render_task: Optional[asyncio.Task] = None
        self._encode_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = PipelineStats()

        logger.info(
            "FramePipeline initialized: buffer_size=%d", buffer_size
        )

    async def start(self, model_id: str) -> None:
        """
        Start the pipeline workers for a given model.

        Launches the render and encode worker tasks that continuously
        process frames from their input queues.

        Args:
            model_id: The loaded model to render with.
        """
        if self._is_running:
            logger.warning("Pipeline already running, stopping first")
            await self.stop()

        self._model_id = model_id
        self._is_running = True

        # Clear queues
        self._clear_queues()

        # Start worker tasks
        self._render_task = asyncio.create_task(
            self._render_worker(), name="render_worker"
        )
        self._encode_task = asyncio.create_task(
            self._encode_worker(), name="encode_worker"
        )

        logger.info("FramePipeline started for model %s", model_id)

    async def stop(self) -> None:
        """
        Stop the pipeline workers and drain queues.

        Sends sentinel values (None) through the pipeline to signal
        workers to exit gracefully.
        """
        if not self._is_running:
            return

        self._is_running = False

        # Send sentinel values to stop workers
        try:
            self._params_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Wait for workers to finish
        if self._render_task:
            try:
                await asyncio.wait_for(self._render_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._render_task.cancel()
            self._render_task = None

        if self._encode_task:
            try:
                await asyncio.wait_for(self._encode_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._encode_task.cancel()
            self._encode_task = None

        self._model_id = None
        logger.info(
            "FramePipeline stopped. Processed %d frames, dropped %d",
            self._stats.frames_completed,
            self._stats.frames_dropped,
        )

    async def submit_params(self, flame_params: dict) -> bool:
        """
        Submit new expression parameters for rendering.

        Non-blocking if the pipeline has room. If the params queue is
        full, drops the oldest params to maintain real-time behavior
        (it's better to skip frames than accumulate latency).

        Args:
            flame_params: FLAME expression parameters dict.

        Returns:
            True if params were accepted, False if dropped.
        """
        if not self._is_running:
            logger.warning("Cannot submit params: pipeline not running")
            return False

        self._stats.frames_submitted += 1

        try:
            self._params_queue.put_nowait(flame_params)
            return True
        except asyncio.QueueFull:
            # Drop oldest params to maintain real-time behavior
            try:
                self._params_queue.get_nowait()
                self._stats.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass

            try:
                self._params_queue.put_nowait(flame_params)
                return True
            except asyncio.QueueFull:
                self._stats.frames_dropped += 1
                return False

    async def get_encoded_frame(self) -> Optional[bytes]:
        """
        Get the next encoded frame from the pipeline.

        Non-blocking: returns None immediately if no frame is ready.
        For real-time use, call this at the target FPS rate.

        Returns:
            Encoded video data bytes, or None if no frame available.
        """
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_encoded_frame_blocking(
        self, timeout: float = 0.1
    ) -> Optional[bytes]:
        """
        Get the next encoded frame, waiting up to timeout seconds.

        Args:
            timeout: Maximum seconds to wait for a frame.

        Returns:
            Encoded video data bytes, or None on timeout.
        """
        try:
            return await asyncio.wait_for(
                self._output_queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def _render_worker(self) -> None:
        """
        Render worker: dequeues FLAME params, renders frames, enqueues
        rendered frames for encoding.

        Runs in a loop until a None sentinel is received or stop() is called.
        """
        logger.debug("Render worker started")

        while self._is_running:
            try:
                params = await asyncio.wait_for(
                    self._params_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if params is None:
                # Propagate sentinel to encode worker
                try:
                    self._render_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                break

            start_time = time.perf_counter()

            try:
                # TODO: Replace with async GPU render:
                # Use asyncio.to_thread() to avoid blocking the event loop
                # while the GPU is rendering (GIL is released during CUDA ops)
                frame = await asyncio.to_thread(
                    self._renderer.render_frame,
                    self._model_id,
                    params,
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._stats.render_times_ms.append(elapsed_ms)

                try:
                    self._render_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    # Drop frame if encode queue is backed up
                    self._stats.frames_dropped += 1
                    logger.debug("Render output dropped (encode queue full)")

            except Exception as e:
                logger.error("Render error: %s", e, exc_info=True)

        logger.debug("Render worker stopped")

    async def _encode_worker(self) -> None:
        """
        Encode worker: dequeues rendered frames, encodes to H.264,
        enqueues encoded data for output.

        In production with NVENC, encoding happens on the GPU and the
        frame data never touches CPU memory (zero-copy pipeline).
        """
        logger.debug("Encode worker started")

        while self._is_running:
            try:
                frame = await asyncio.wait_for(
                    self._render_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if frame is None:
                break

            start_time = time.perf_counter()

            try:
                # TODO: Replace with async NVENC encode
                encoded = await asyncio.to_thread(
                    self._encoder.encode_frame, frame
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._stats.encode_times_ms.append(elapsed_ms)

                total_ms = self._stats.render_times_ms[-1] + elapsed_ms if self._stats.render_times_ms else elapsed_ms
                self._stats.total_times_ms.append(total_ms)
                self._stats.frames_completed += 1

                try:
                    self._output_queue.put_nowait(encoded)
                except asyncio.QueueFull:
                    # Drop oldest encoded frame
                    try:
                        self._output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._output_queue.put_nowait(encoded)
                    self._stats.frames_dropped += 1

            except Exception as e:
                logger.error("Encode error: %s", e, exc_info=True)

        logger.debug("Encode worker stopped")

    def _clear_queues(self) -> None:
        """Drain all pipeline queues."""
        for q in [self._params_queue, self._render_queue, self._output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def get_stats(self) -> dict:
        """
        Return pipeline performance statistics.

        Returns:
            dict with keys: is_running, model_id, buffer_size,
            frames_submitted, frames_completed, frames_dropped,
            avg_render_ms, avg_encode_ms, avg_total_ms,
            params_queue_depth, render_queue_depth, output_queue_depth,
            estimated_fps.
        """
        avg_total = self._stats.avg_total_ms()

        return {
            "is_running": self._is_running,
            "model_id": self._model_id,
            "buffer_size": self._buffer_size,
            "frames_submitted": self._stats.frames_submitted,
            "frames_completed": self._stats.frames_completed,
            "frames_dropped": self._stats.frames_dropped,
            "drop_rate": round(
                self._stats.frames_dropped / max(1, self._stats.frames_submitted), 3
            ),
            "avg_render_ms": round(self._stats.avg_render_ms(), 2),
            "avg_encode_ms": round(self._stats.avg_encode_ms(), 2),
            "avg_total_ms": round(avg_total, 2),
            "estimated_fps": round(1000 / avg_total, 1) if avg_total > 0 else 0,
            "params_queue_depth": self._params_queue.qsize(),
            "render_queue_depth": self._render_queue.qsize(),
            "output_queue_depth": self._output_queue.qsize(),
        }
