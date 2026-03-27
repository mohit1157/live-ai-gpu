"""
LRU Model Cache for GPU VRAM management.

Keeps the N most recently used avatar models loaded in GPU VRAM for instant
switching between users/avatars. Evicts least-recently-used models when the
VRAM budget is exhausted.

Design rationale:
- On an A10G (24GB VRAM), a typical Gaussian avatar model uses ~1-2GB.
  Keeping 5 models loaded means we can serve 5 concurrent users without
  any model loading latency (~2-3s cold load from S3).
- The cache tracks VRAM usage per model and enforces a configurable cap
  to prevent OOM conditions.
- Thread-safe via threading.Lock for concurrent request handling.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached model with metadata."""
    model: Any
    vram_size_gb: float
    loaded_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


class ModelCache:
    """
    LRU cache for loaded avatar models in GPU VRAM.
    Keeps the N most recently used models loaded for instant switching.
    Evicts least-recently-used models when VRAM limit is reached.

    Thread-safe: all public methods are protected by a lock to support
    concurrent FastAPI request handlers.

    Typical configuration for AWS g5.xlarge (A10G, 24GB VRAM):
    - max_models=5 (5 avatars loaded simultaneously)
    - max_vram_gb=16.0 (reserve 8GB for rendering scratch space)
    """

    def __init__(self, max_models: int = 5, max_vram_gb: float = 16.0) -> None:
        self._max_models = max_models
        self._max_vram_gb = max_vram_gb
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._total_vram_gb: float = 0.0

        # Stats
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        logger.info(
            "ModelCache initialized: max_models=%d, max_vram_gb=%.1f",
            max_models,
            max_vram_gb,
        )

    def get(self, model_id: str) -> Optional[Any]:
        """
        Retrieve a cached model by ID.

        Returns the model if found (cache hit), or None (cache miss).
        Moves the entry to the end of the LRU order on hit.
        """
        with self._lock:
            if model_id not in self._cache:
                self._misses += 1
                return None

            self._hits += 1
            entry = self._cache[model_id]
            entry.last_accessed = time.time()
            entry.access_count += 1

            # Move to end (most recently used)
            self._cache.move_to_end(model_id)

            return entry.model

    def put(self, model_id: str, model: Any, vram_size_gb: float) -> None:
        """
        Insert or update a model in the cache.

        If inserting would exceed the VRAM budget or model count limit,
        evicts LRU entries until there is room.

        Args:
            model_id: Unique identifier for the model.
            model: The loaded model object (opaque to the cache).
            vram_size_gb: Estimated VRAM footprint of this model.
        """
        with self._lock:
            # If already cached, remove old entry first
            if model_id in self._cache:
                old_entry = self._cache.pop(model_id)
                self._total_vram_gb -= old_entry.vram_size_gb

            # Evict until we have room
            while (
                len(self._cache) >= self._max_models
                or self._total_vram_gb + vram_size_gb > self._max_vram_gb
            ) and self._cache:
                self._evict_lru_internal()

            # Insert new entry
            self._cache[model_id] = CacheEntry(
                model=model,
                vram_size_gb=vram_size_gb,
            )
            self._total_vram_gb += vram_size_gb

            logger.info(
                "Cached model %s (%.2f GB VRAM). Total: %d models, %.2f GB",
                model_id,
                vram_size_gb,
                len(self._cache),
                self._total_vram_gb,
            )

    def evict_lru(self) -> Optional[str]:
        """
        Manually evict the least recently used model.

        Returns the evicted model_id or None if cache is empty.
        """
        with self._lock:
            return self._evict_lru_internal()

    def _evict_lru_internal(self) -> Optional[str]:
        """Evict LRU entry (caller must hold lock)."""
        if not self._cache:
            return None

        model_id, entry = self._cache.popitem(last=False)
        self._total_vram_gb -= entry.vram_size_gb
        self._evictions += 1

        logger.info(
            "Evicted model %s (%.2f GB freed). Total: %d models, %.2f GB",
            model_id,
            entry.vram_size_gb,
            len(self._cache),
            self._total_vram_gb,
        )

        # TODO: Replace with actual GPU memory cleanup
        # torch.cuda.empty_cache() or explicit tensor deletion
        del entry.model

        return model_id

    def contains(self, model_id: str) -> bool:
        """Check if a model is in the cache without updating LRU order."""
        with self._lock:
            return model_id in self._cache

    def get_stats(self) -> dict:
        """
        Return cache statistics.

        Returns:
            dict with keys: cached_models, max_models, vram_used_gb,
            max_vram_gb, hits, misses, hit_rate, evictions, model_ids.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "cached_models": len(self._cache),
                "max_models": self._max_models,
                "vram_used_gb": round(self._total_vram_gb, 2),
                "max_vram_gb": self._max_vram_gb,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "evictions": self._evictions,
                "model_ids": list(self._cache.keys()),
            }

    def clear(self) -> None:
        """Remove all models from the cache."""
        with self._lock:
            self._cache.clear()
            self._total_vram_gb = 0.0
            logger.info("ModelCache cleared")
