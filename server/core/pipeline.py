"""
Async pipeline — wires all components together via asyncio.Queue channels.

Each pipeline stage runs as an independent asyncio coroutine on the same
event loop. CPU-bound inference executes in ThreadPoolExecutors (one per model)
via run_in_executor, keeping the event loop responsive.

Queue topology:
  audio_chunk_q (drop-oldest on full)
    → _vad_coroutine
      → speech_segment_q (block on full)
        → _asr_coroutine (emits partials + final)
          → IncrementalTranslationState
            → (translate) → (TTS) → audio_output_q

Each session (WebSocket connection) gets one AsyncPipeline instance.
All models are shared across sessions via the ComponentBundle.
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import structlog

from server.audio.ringbuffer import RingBuffer
from server.audio.vad import SpeechSegment, VADSegmenter
from server.asr.engine import ASRResult
from server.core.config import Config
from server.core.incremental import IncrementalASRState, IncrementalTranslationState
from server.core.metrics import MetricsCollector
from server.core.scheduler import ComponentBundle

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16_000


# ── Event types emitted by the pipeline ────────────────────────────────────

@dataclass
class VADEvent:
    state: str   # "speech_start" | "speech_end"
    session_id: str

    def to_dict(self) -> dict:
        return {"type": "vad_event", "state": self.state, "session_id": self.session_id}


@dataclass
class PartialTranscriptEvent:
    text: str
    new_words: list[str]
    confidence: float
    session_id: str

    def to_dict(self) -> dict:
        return {
            "type": "partial_transcript",
            "text": self.text,
            "new_words": self.new_words,
            "confidence": round(self.confidence, 3),
            "session_id": self.session_id,
        }


@dataclass
class FinalTranscriptEvent:
    text: str
    translated: str
    confidence: float
    session_id: str
    asr_ms: float
    translation_ms: float

    def to_dict(self) -> dict:
        return {
            "type": "final_transcript",
            "text": self.text,
            "translated": self.translated,
            "confidence": round(self.confidence, 3),
            "session_id": self.session_id,
            "asr_ms": round(self.asr_ms, 1),
            "translation_ms": round(self.translation_ms, 1),
        }


@dataclass
class AudioChunkEvent:
    pcm_bytes: bytes   # Int16 LE PCM
    session_id: str
    is_utterance_end: bool = False


@dataclass
class PipelineErrorEvent:
    stage: str
    message: str
    recoverable: bool
    session_id: str

    def to_dict(self) -> dict:
        return {
            "type": "error",
            "stage": self.stage,
            "message": self.message,
            "recoverable": self.recoverable,
            "session_id": self.session_id,
        }


PipelineEvent = VADEvent | PartialTranscriptEvent | FinalTranscriptEvent | AudioChunkEvent | PipelineErrorEvent


class AsyncPipeline:
    """One pipeline instance per WebSocket session.

    Components (models) are shared across sessions via ComponentBundle.
    Each instance has its own asyncio.Queue set and IncrementalTranslationState.
    """

    def __init__(
        self,
        session_id: str,
        bundle: ComponentBundle,
        config: Config,
        metrics: MetricsCollector,
    ) -> None:
        self._session_id = session_id
        self._bundle = bundle
        self._config = config
        self._metrics = metrics

        q = config.pipeline.queues
        # Drop-oldest audio queue — stale audio is worse than dropped audio
        self._audio_chunk_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=q.audio_chunk_maxsize)
        # Blocking speech segment queue — dropping a segment corrupts incremental state
        self._speech_segment_q: asyncio.Queue[Optional[SpeechSegment]] = asyncio.Queue(maxsize=q.speech_segment_maxsize)
        # Audio output queue consumed by the WebSocket sender
        self._audio_output_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=q.tts_audio_maxsize)
        # Event queue for all JSON events (transcripts, VAD, metrics)
        self._event_q: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=50)

        # Per-utterance ring buffer for partial transcripts during accumulation
        self._audio_ringbuf = RingBuffer(
            capacity_samples=int(_SAMPLE_RATE * 45),  # 45 seconds max
            dtype=np.float32,
        )

        # VAD segmenter: each session gets its own copy of the segmenter state
        # (the underlying model object is shared and thread-safe for read-only inference)
        self._vad_segmenter = VADSegmenter(
            bundle.vad_engine,
            config.vad,
        )

        self._asr_state = IncrementalASRState(
            stability_window=config.asr.stability_window,
        )
        self._incremental_state = IncrementalTranslationState(
            asr_state=self._asr_state,
            translator=bundle.translator,
            tts=bundle.tts,
            audio_output_queue=self._audio_output_q,
            source_lang=config.translation.source_lang,
            target_lang=config.translation.target_lang,
            context_window_chars=config.translation.context_window_chars,
            min_commit_words=config.pipeline.incremental.min_commit_words,
        )

        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._muted = False
        self._previous_asr_text: str = ""

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._vad_coroutine(), name=f"vad-{self._session_id[:8]}"),
            asyncio.create_task(self._asr_coroutine(), name=f"asr-{self._session_id[:8]}"),
            asyncio.create_task(self._audio_forwarder(), name=f"fwd-{self._session_id[:8]}"),
        ]
        logger.info("Pipeline started", session_id=self._session_id)

    async def stop(self) -> None:
        self._running = False
        # Unblock queue consumers by sending sentinel values
        try:
            self._audio_chunk_q.put_nowait(None)
        except asyncio.QueueFull:
            pass
        try:
            self._speech_segment_q.put_nowait(None)
        except asyncio.QueueFull:
            pass

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("Pipeline stopped", session_id=self._session_id)

    # ── Audio Input ─────────────────────────────────────────────────────────

    async def feed_audio(self, pcm_bytes: bytes) -> None:
        """Receive raw Int16 LE PCM bytes from WebSocket and queue for VAD."""
        if self._muted or not self._running:
            return

        # Convert to float32
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0

        # Write to ring buffer for incremental ASR partials
        self._audio_ringbuf.write(audio_f32)

        # Apply noise reduction (sync, fast)
        if self._config.noise_reduction.enabled:
            audio_f32 = self._bundle.noise_reducer.process(audio_f32)

        # Queue for VAD — drop oldest if full (prefer fresh audio)
        chunk_size = self._config.vad.window_size_samples
        for i in range(0, len(audio_f32), chunk_size):
            chunk = audio_f32[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            await self._put_dropping_oldest(self._audio_chunk_q, chunk.tobytes())

    async def _put_dropping_oldest(
        self,
        q: asyncio.Queue,
        item: Any,
    ) -> None:
        """Put item in queue; drop the oldest entry if queue is full."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await q.put(item)

    # ── VAD Coroutine ───────────────────────────────────────────────────────

    async def _vad_coroutine(self) -> None:
        """Consume audio chunks, emit SpeechSegments on end-of-speech."""
        in_speech = False
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                item = await asyncio.wait_for(self._audio_chunk_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if item is None:
                break

            chunk_bytes: bytes = item
            chunk = np.frombuffer(chunk_bytes, dtype=np.float32)

            segments = await loop.run_in_executor(
                None,
                self._vad_segmenter.process_chunk,
                chunk,
                self._session_id,
            )

            was_in_speech = in_speech
            in_speech = self._vad_segmenter.is_in_speech

            if not was_in_speech and in_speech:
                self._audio_ringbuf.clear()
                await self._event_q.put(VADEvent("speech_start", self._session_id))

            for seg in segments:
                if not was_in_speech and not in_speech:
                    # VAD detected a very short segment — still emit
                    pass
                await self._event_q.put(VADEvent("speech_end", self._session_id))
                try:
                    await asyncio.wait_for(
                        self._speech_segment_q.put(seg),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Speech segment queue full — dropping segment",
                        session_id=self._session_id,
                    )
                    await self._event_q.put(
                        PipelineErrorEvent("vad", "Segment queue full — segment dropped", True, self._session_id)
                    )

    # ── ASR Coroutine ───────────────────────────────────────────────────────

    async def _asr_coroutine(self) -> None:
        """Consume SpeechSegments, run ASR, feed IncrementalTranslationState."""
        while self._running:
            try:
                seg = await asyncio.wait_for(self._speech_segment_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if seg is None:
                break

            t_asr_start = time.perf_counter()
            try:
                result: ASRResult = await self._bundle.asr.transcribe(
                    seg.audio,
                    session_id=self._session_id,
                    previous_text=self._previous_asr_text,
                )
            except Exception as exc:
                logger.error("ASR failed", error=str(exc), session_id=self._session_id)
                await self._event_q.put(
                    PipelineErrorEvent("asr", str(exc), True, self._session_id)
                )
                self._incremental_state.reset()
                continue

            asr_latency_ms = (time.perf_counter() - t_asr_start) * 1000
            self._metrics.record_asr(asr_latency_ms)

            # Update previous text for next utterance conditioning
            if result.text:
                self._previous_asr_text = result.text

            # Emit partial transcript event
            partial_words = result.word_list
            await self._event_q.put(
                PartialTranscriptEvent(
                    text=result.text,
                    new_words=partial_words,
                    confidence=result.confidence,
                    session_id=self._session_id,
                )
            )

            # Drive the incremental translation + TTS state machine
            t_translation_start = time.perf_counter()
            try:
                if self._config.pipeline.incremental.enabled:
                    await self._incremental_state.on_final_words(partial_words)
                else:
                    # Non-incremental: translate the full utterance at once
                    translation_result = await self._bundle.translator.translate(
                        result.text,
                        self._config.translation.source_lang,
                        self._config.translation.target_lang,
                    )
                    if translation_result.translated_text:
                        tts_result = await self._bundle.tts.synthesize(
                            translation_result.translated_text
                        )
                        import numpy as _np
                        int16 = (_np.array(tts_result.audio) * 32767).clip(-32768, 32767).astype(_np.int16)
                        await self._audio_output_q.put(int16.tobytes())
                        await self._audio_output_q.put(b"")

            except Exception as exc:
                logger.error("Translation/TTS failed", error=str(exc), session_id=self._session_id)
                await self._event_q.put(
                    PipelineErrorEvent("translation", str(exc), True, self._session_id)
                )
                self._incremental_state.reset()
                continue

            translation_latency_ms = (time.perf_counter() - t_translation_start) * 1000
            self._metrics.record_translation(translation_latency_ms)
            self._metrics.record_utterance(len(partial_words))

            await self._event_q.put(
                FinalTranscriptEvent(
                    text=result.text,
                    translated=self._incremental_state.committed_translation,
                    confidence=result.confidence,
                    session_id=self._session_id,
                    asr_ms=asr_latency_ms,
                    translation_ms=translation_latency_ms,
                )
            )

            self._incremental_state.reset()

    # ── Audio Forwarder ─────────────────────────────────────────────────────

    async def _audio_forwarder(self) -> None:
        """Move PCM bytes from audio_output_q to event_q as AudioChunkEvents."""
        while self._running:
            try:
                pcm = await asyncio.wait_for(self._audio_output_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            is_end = pcm == b""
            await self._event_q.put(
                AudioChunkEvent(
                    pcm_bytes=pcm,
                    session_id=self._session_id,
                    is_utterance_end=is_end,
                )
            )

    # ── Event stream ────────────────────────────────────────────────────────

    async def event_stream(self) -> AsyncIterator[PipelineEvent]:
        """Yield pipeline events (transcripts, audio, errors) as they arrive."""
        while self._running or not self._event_q.empty():
            try:
                event = await asyncio.wait_for(self._event_q.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

    # ── Control ─────────────────────────────────────────────────────────────

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def update_target_lang(self, target_lang: str) -> None:
        self._incremental_state._target_lang = target_lang

    @property
    def session_id(self) -> str:
        return self._session_id


class PipelineFactory:
    """Creates and tracks AsyncPipeline sessions.

    Enforces the max_concurrent_sessions limit by rejecting new connections
    when all slots are occupied.
    """

    def __init__(
        self,
        bundle: ComponentBundle,
        config: Config,
        metrics: MetricsCollector,
    ) -> None:
        self._bundle = bundle
        self._config = config
        self._metrics = metrics
        self._sessions: dict[str, AsyncPipeline] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str) -> AsyncPipeline:
        async with self._lock:
            if len(self._sessions) >= self._config.server.max_concurrent_sessions:
                raise RuntimeError(
                    f"Maximum concurrent sessions ({self._config.server.max_concurrent_sessions}) reached"
                )
            pipeline = AsyncPipeline(session_id, self._bundle, self._config, self._metrics)
            self._sessions[session_id] = pipeline

        await pipeline.start()
        return pipeline

    async def destroy_session(self, session_id: str) -> None:
        async with self._lock:
            pipeline = self._sessions.pop(session_id, None)

        if pipeline:
            await pipeline.stop()

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
