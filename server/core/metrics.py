"""
Prometheus metrics and rolling latency statistics.

Exposes a /metrics endpoint (via prometheus_client) and maintains
a sliding window for real-time latency percentile computation.
Also provides a background GPU poller that updates VRAM/utilization gauges.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY

    _REGISTRY = REGISTRY
    _prometheus_available = True
except ImportError:
    _prometheus_available = False
    logger.warning("prometheus_client not available — metrics disabled")


@dataclass
class LatencySnapshot:
    """Rolling window latency statistics for one pipeline stage."""

    stage: str
    window: collections.deque = field(default_factory=lambda: collections.deque(maxlen=100))

    def record(self, latency_ms: float) -> None:
        self.window.append(latency_ms)

    @property
    def p50_ms(self) -> float:
        if not self.window:
            return 0.0
        s = sorted(self.window)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.window:
            return 0.0
        s = sorted(self.window)
        return s[int(len(s) * 0.95)]

    @property
    def avg_ms(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "avg_ms": round(self.avg_ms, 1),
            "samples": len(self.window),
        }


@dataclass
class PipelineMetricsSnapshot:
    """Point-in-time snapshot of all pipeline metrics — sent over WebSocket."""

    asr_ms: float
    translation_ms: float
    tts_ms: float
    total_ms: float
    gpu_util_pct: float
    gpu_mem_used_gb: float
    gpu_mem_total_gb: float
    utterances_processed: int
    words_per_second: float

    def to_dict(self) -> dict:
        return {
            "type": "metrics",
            "asr_ms": round(self.asr_ms, 1),
            "translation_ms": round(self.translation_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "gpu_mem_used_gb": round(self.gpu_mem_used_gb, 2),
            "gpu_mem_total_gb": round(self.gpu_mem_total_gb, 2),
            "utterances": self.utterances_processed,
            "wps": round(self.words_per_second, 1),
        }


class MetricsCollector:
    """Collects pipeline latency stats and GPU telemetry.

    Prometheus gauges/histograms are registered if prometheus_client is available.
    Rolling windows are always maintained regardless of Prometheus.
    """

    def __init__(self, window_size: int = 100, device_index: int = 0) -> None:
        self._window_size = window_size
        self._device_index = device_index

        self.vad = LatencySnapshot("vad")
        self.asr = LatencySnapshot("asr")
        self.translation = LatencySnapshot("translation")
        self.tts = LatencySnapshot("tts")
        self.e2e = LatencySnapshot("e2e")

        self._utterances: int = 0
        self._total_words: int = 0
        self._session_start = time.monotonic()

        self._gpu_util: float = 0.0
        self._gpu_mem_used_gb: float = 0.0
        self._gpu_mem_total_gb: float = 0.0

        self._lock = threading.Lock()
        self._gpu_poll_task: Optional[asyncio.Task] = None

        if _prometheus_available:
            self._setup_prometheus()

    def _setup_prometheus(self) -> None:
        self._p_asr_hist = Histogram(
            "translator_asr_latency_ms",
            "ASR inference latency in milliseconds",
            buckets=[50, 100, 150, 200, 250, 300, 400, 500, 1000],
        )
        self._p_translation_hist = Histogram(
            "translator_translation_latency_ms",
            "Translation latency in milliseconds",
            buckets=[10, 25, 50, 75, 100, 150, 200],
        )
        self._p_tts_hist = Histogram(
            "translator_tts_latency_ms",
            "TTS latency in milliseconds",
            buckets=[20, 40, 60, 100, 150, 200, 300],
        )
        self._p_e2e_hist = Histogram(
            "translator_e2e_latency_ms",
            "End-to-end latency (capture to audio start) in milliseconds",
            buckets=[300, 500, 700, 900, 1200, 1500],
        )
        self._p_gpu_util = Gauge("translator_gpu_utilization_pct", "GPU utilization percent")
        self._p_gpu_mem = Gauge("translator_gpu_mem_used_gb", "GPU memory used in GB")
        self._p_utterances = Counter("translator_utterances_total", "Total utterances processed")

    def record_asr(self, latency_ms: float) -> None:
        self.asr.record(latency_ms)
        if _prometheus_available:
            self._p_asr_hist.observe(latency_ms)

    def record_translation(self, latency_ms: float) -> None:
        self.translation.record(latency_ms)
        if _prometheus_available:
            self._p_translation_hist.observe(latency_ms)

    def record_tts(self, latency_ms: float) -> None:
        self.tts.record(latency_ms)
        if _prometheus_available:
            self._p_tts_hist.observe(latency_ms)

    def record_e2e(self, latency_ms: float) -> None:
        self.e2e.record(latency_ms)
        if _prometheus_available:
            self._p_e2e_hist.observe(latency_ms)

    def record_utterance(self, word_count: int) -> None:
        with self._lock:
            self._utterances += 1
            self._total_words += word_count
        if _prometheus_available:
            self._p_utterances.inc()

    def snapshot(self) -> PipelineMetricsSnapshot:
        with self._lock:
            elapsed = max(time.monotonic() - self._session_start, 1.0)
            wps = self._total_words / elapsed

        return PipelineMetricsSnapshot(
            asr_ms=self.asr.avg_ms,
            translation_ms=self.translation.avg_ms,
            tts_ms=self.tts.avg_ms,
            total_ms=self.e2e.avg_ms,
            gpu_util_pct=self._gpu_util,
            gpu_mem_used_gb=self._gpu_mem_used_gb,
            gpu_mem_total_gb=self._gpu_mem_total_gb,
            utterances_processed=self._utterances,
            words_per_second=wps,
        )

    # ── GPU polling ─────────────────────────────────────────────────────────

    async def start_gpu_poller(self, interval_s: float = 2.0) -> None:
        """Start a background task that polls GPU stats every interval_s seconds."""
        self._gpu_poll_task = asyncio.create_task(
            self._gpu_poll_loop(interval_s),
            name="gpu_poller",
        )

    async def stop_gpu_poller(self) -> None:
        if self._gpu_poll_task:
            self._gpu_poll_task.cancel()
            try:
                await self._gpu_poll_task
            except asyncio.CancelledError:
                pass

    async def _gpu_poll_loop(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                self._poll_gpu_sync()
            except Exception as exc:
                logger.debug("GPU poll failed", error=str(exc))

    def _poll_gpu_sync(self) -> None:
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._gpu_util = float(util.gpu)

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._gpu_mem_used_gb = mem.used / 1024**3
            self._gpu_mem_total_gb = mem.total / 1024**3

            if _prometheus_available:
                self._p_gpu_util.set(self._gpu_util)
                self._p_gpu_mem.set(self._gpu_mem_used_gb)
        except Exception:
            # pynvml not installed or no GPU — use torch fallback
            try:
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(self._device_index)
                    self._gpu_mem_total_gb = total / 1024**3
                    self._gpu_mem_used_gb = (total - free) / 1024**3
            except Exception:
                pass
