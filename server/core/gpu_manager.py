"""
GPU memory manager.

Tracks VRAM allocation across all loaded models, enforces an allocation
guard to prevent OOM, and provides a live VRAM reading for metrics.

The VRAM budget for RTX 4070 Ti (12 GB):

  Faster-Whisper large-v3 INT8   ~1.5 GB
  SeamlessM4T v2 large FP16      ~4.5 GB
  Silero VAD ONNX (CPU)          ~0.0 GB
  CUDA runtime + activations     ~1.5 GB
  Safety margin                  ~1.0 GB
  ──────────────────────────────────────
  Total used                     ~8.5 GB
  Free headroom                  ~3.5 GB
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import ClassVar

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelMemoryProfile:
    name: str
    estimated_vram_gb: float
    priority: int   # 1 = critical (always loaded), 2 = high, 3 = can offload


class GPUMemoryManager:
    """Tracks per-model VRAM allocation and guards against OOM.

    Thread-safe: model registration may happen from multiple threads
    (one per inference ThreadPoolExecutor) at startup.
    """

    PROFILES: ClassVar[dict[str, ModelMemoryProfile]] = {
        # Keys are built from the configured compute_type, so they must match the
        # strings faster-whisper actually accepts ("float16", not "fp16") --
        # otherwise register_loaded() silently records 0 GB for the model.
        "faster_whisper_large_v3_int8":         ModelMemoryProfile("faster_whisper_large_v3_int8",         1.5, 1),
        "faster_whisper_large_v3_int8_float16": ModelMemoryProfile("faster_whisper_large_v3_int8_float16", 2.0, 1),
        "faster_whisper_large_v3_float16":      ModelMemoryProfile("faster_whisper_large_v3_float16",      3.0, 1),
        "faster_whisper_large_v3_float32":      ModelMemoryProfile("faster_whisper_large_v3_float32",      6.0, 1),
        "faster_whisper_large_v3_fp16":         ModelMemoryProfile("faster_whisper_large_v3_fp16",         3.0, 1),
        "seamless_m4t_v2_large_fp16":     ModelMemoryProfile("seamless_m4t_v2_large_fp16",     4.5, 1),
        "nllb_200_distilled_1_3b_fp16":   ModelMemoryProfile("nllb_200_distilled_1_3b_fp16",   2.6, 2),
        "silero_vad":                     ModelMemoryProfile("silero_vad",                     0.05, 1),
        "piper_tts_cpu":                  ModelMemoryProfile("piper_tts_cpu",                  0.0, 1),
        "deepfilternet3_gpu":             ModelMemoryProfile("deepfilternet3_gpu",              0.3, 2),
    }

    def __init__(self, device_index: int = 0, safety_margin_gb: float = 1.0) -> None:
        self._device_index = device_index
        self._safety_margin = safety_margin_gb
        self._loaded_models: dict[str, float] = {}   # name → allocated GB
        self._lock = threading.Lock()

    # ── Queries ─────────────────────────────────────────────────────────────

    @property
    def total_gb(self) -> float:
        try:
            import torch
            total, _ = torch.cuda.mem_get_info(self._device_index)
            return total / 1024**3
        except Exception:
            return 0.0

    @property
    def available_gb(self) -> float:
        """Live VRAM available — from torch.cuda.mem_get_info()."""
        try:
            import torch
            _, free = torch.cuda.mem_get_info(self._device_index)
            return free / 1024**3
        except Exception:
            return 0.0

    @property
    def allocated_gb(self) -> float:
        with self._lock:
            return sum(self._loaded_models.values())

    @property
    def effective_available_gb(self) -> float:
        """Available VRAM minus the safety margin."""
        return max(0.0, self.available_gb - self._safety_margin)

    # ── Allocation Guards ───────────────────────────────────────────────────

    def can_allocate(self, model_name: str) -> bool:
        """Return True if VRAM is sufficient to load model_name."""
        profile = self.PROFILES.get(model_name)
        if profile is None:
            return True   # Unknown model — optimistically allow
        return self.effective_available_gb >= profile.estimated_vram_gb

    def assert_can_allocate(self, model_name: str) -> None:
        """Raise RuntimeError if VRAM is insufficient."""
        if not self.can_allocate(model_name):
            profile = self.PROFILES.get(model_name)
            needed = profile.estimated_vram_gb if profile else 0
            raise RuntimeError(
                f"Insufficient VRAM for {model_name}: "
                f"need {needed:.1f} GB, have {self.effective_available_gb:.1f} GB free "
                f"(total {self.total_gb:.1f} GB, {self.available_gb:.1f} GB available)"
            )

    # ── Registration ────────────────────────────────────────────────────────

    def register_loaded(self, model_name: str, actual_gb: float | None = None) -> None:
        """Record that model_name has been loaded into VRAM."""
        profile = self.PROFILES.get(model_name)
        if profile is None and actual_gb is None:
            logger.warning(
                "No VRAM profile for model -- accounting it as 0 GB. "
                "Add a PROFILES entry, or the allocation guards will "
                "under-count this model.",
                model=model_name,
                known_profiles=sorted(self.PROFILES),
            )
        estimated = actual_gb or (profile.estimated_vram_gb if profile else 0.0)

        with self._lock:
            self._loaded_models[model_name] = estimated

        logger.info(
            "Model loaded",
            model=model_name,
            vram_gb=round(estimated, 2),
            total_allocated_gb=round(self.allocated_gb, 2),
            available_gb=round(self.available_gb, 2),
        )

    def register_unloaded(self, model_name: str) -> None:
        """Record that model_name has been removed from VRAM."""
        with self._lock:
            freed = self._loaded_models.pop(model_name, 0.0)

        logger.info(
            "Model unloaded",
            model=model_name,
            freed_gb=round(freed, 2),
            available_gb=round(self.available_gb, 2),
        )

    # ── OOM Recovery ────────────────────────────────────────────────────────

    def handle_oom(self, component: str) -> None:
        """Called when a torch.cuda.OutOfMemoryError is caught during inference.

        Clears the CUDA allocator caches and logs the event.
        If still OOM after clearing, the caller should attempt to unload
        lower-priority models via the component registry.
        """
        logger.error("CUDA OOM during %s — clearing cache", component)
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self._device_index)
            logger.info(
                "Cache cleared after OOM",
                available_gb=round(self.available_gb, 2),
            )
        except Exception as exc:
            logger.error("Failed to clear CUDA cache", error=str(exc))

    # ── Logging ─────────────────────────────────────────────────────────────

    def log_status(self) -> None:
        """Log current VRAM state at INFO level."""
        with self._lock:
            loaded = dict(self._loaded_models)

        logger.info(
            "VRAM status",
            loaded_models=list(loaded.keys()),
            allocated_gb=round(sum(loaded.values()), 2),
            available_gb=round(self.available_gb, 2),
            total_gb=round(self.total_gb, 2),
        )

    def to_dict(self) -> dict:
        with self._lock:
            loaded = dict(self._loaded_models)
        return {
            "loaded_models": loaded,
            "allocated_gb": round(sum(loaded.values()), 2),
            "available_gb": round(self.available_gb, 2),
            "total_gb": round(self.total_gb, 2),
        }
