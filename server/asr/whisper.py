"""
Faster-Whisper ASR backend — primary engine for Nepali transcription.

Uses CTranslate2 INT8 quantization for ~2× speedup over FP16 PyTorch Whisper
with negligible WER impact on Nepali (large-v3 model capacity dominates).

Streaming strategy:
  - External VAD (SileroVAD) hands us complete speech segments.
  - We transcribe each segment in a dedicated ThreadPoolExecutor worker to
    avoid blocking the asyncio event loop.
  - The transcribe_with_partials() method additionally supports emitting
    incremental results while the VAD segment is still accumulating,
    by snapshotting the ring buffer every partial_interval_s seconds.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable

import numpy as np
import structlog

from server.asr.engine import ASREngine, ASRResult, WordTimestamp
from server.audio.ringbuffer import RingBuffer
from server.core.config import ASRConfig

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16_000


class FasterWhisperASR(ASREngine):
    """CTranslate2-accelerated Whisper for real-time Nepali ASR.

    The model is loaded once at startup. All inference runs in a single-threaded
    ThreadPoolExecutor so CUDA context stays on one OS thread throughout.
    """

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._loaded = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def load(self) -> None:
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._config.model_path,
            device=self._config.device,
            compute_type=self._config.compute_type,
            num_workers=1,
            download_root=None,
        )

        # Warm-up pass — allocates CUDA memory; subsequent calls are faster
        silence = np.zeros(int(_SAMPLE_RATE * 1.0), dtype=np.float32)
        list(self._model.transcribe(silence, language=self._config.language)[0])

        self._loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Faster-Whisper loaded",
            model=self._config.model,
            compute_type=self._config.compute_type,
            device=self._config.device,
            load_ms=round(elapsed),
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("Faster-Whisper unloaded")

    # ── Inference ───────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio: np.ndarray,
        session_id: str = "",
        previous_text: str = "",
    ) -> ASRResult:
        """Transcribe audio in the dedicated executor thread."""
        if not self._loaded or self._model is None:
            raise RuntimeError("FasterWhisperASR not loaded — call load() first")

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        result = await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio,
            previous_text,
            session_id,
        )
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        previous_text: str,
        session_id: str,
    ) -> ASRResult:
        """Synchronous transcription — runs inside the executor thread."""
        assert self._model is not None
        cfg = self._config

        audio_f32 = audio.astype(np.float32)
        duration_s = len(audio_f32) / _SAMPLE_RATE

        segments, info = self._model.transcribe(
            audio_f32,
            language=cfg.language if cfg.language else None,
            beam_size=cfg.beam_size,
            best_of=cfg.best_of,
            patience=cfg.patience,
            temperature=cfg.temperature,
            word_timestamps=cfg.word_timestamps,
            condition_on_previous_text=cfg.condition_on_previous_text,
            initial_prompt=previous_text if previous_text else (cfg.initial_prompt or None),
            vad_filter=False,   # upstream SileroVAD handles this
        )

        # Materialize the lazy segment generator
        all_words: list[WordTimestamp] = []
        all_text_parts: list[str] = []

        for seg in segments:
            all_text_parts.append(seg.text)
            if cfg.word_timestamps and seg.words:
                for w in seg.words:
                    all_words.append(
                        WordTimestamp(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

        full_text = "".join(all_text_parts).strip()
        confidence = (
            float(np.mean([w.probability for w in all_words]))
            if all_words
            else 0.0
        )

        return ASRResult(
            text=full_text,
            is_final=True,
            confidence=confidence,
            words=all_words,
            language=info.language,
            latency_ms=0.0,   # caller fills this in
            audio_duration_s=duration_s,
            session_id=session_id,
        )

    async def transcribe_with_partials(
        self,
        audio_buffer: RingBuffer,
        session_id: str,
        on_partial: Callable[[ASRResult], Awaitable[None]],
        stop_event: asyncio.Event,
        previous_text: str = "",
    ) -> None:
        """Emit partial transcripts while VAD is still accumulating audio.

        Runs until stop_event is set (VAD detected end-of-speech).
        Every partial_interval_s seconds, snapshots the ring buffer and
        runs Whisper to produce a partial result. The caller's on_partial
        callback receives each partial result.

        This method does NOT return the final result — the caller should
        separately call transcribe() on the complete audio after stop_event fires.
        """
        interval = self._config.partial_interval_s
        prev = ""

        while not stop_event.is_set():
            # Wait for the interval or until stop is requested
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.ensure_future(stop_event.wait())),
                    timeout=interval,
                )
                break  # stop_event fired
            except asyncio.TimeoutError:
                pass   # interval elapsed — emit partial

            snapshot = audio_buffer.snapshot()
            if len(snapshot) < int(_SAMPLE_RATE * 0.5):
                continue  # not enough audio yet

            loop = asyncio.get_running_loop()
            partial = await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                snapshot,
                prev,
                session_id,
            )
            partial.is_final = False

            if partial.text and partial.text != prev:
                await on_partial(partial)
                prev = partial.text

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_gb(self) -> float:
        """Estimate based on model size and compute type."""
        base = {
            "large-v3": 3.0,
            "large-v2": 3.0,
            "medium": 1.5,
            "small": 0.5,
            "base": 0.1,
            "tiny": 0.04,
        }
        factor = 0.5 if self._config.compute_type == "int8" else 1.0
        gb = base.get(self._config.model, 3.0) * factor
        return gb if self._loaded else 0.0

    @property
    def model_name(self) -> str:
        return f"faster-whisper-{self._config.model}-{self._config.compute_type}"
