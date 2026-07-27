"""
Voice Activity Detection.

Primary: Silero VAD v5 via ONNX Runtime.
The VADSegmenter state machine wraps any VADEngine and converts a raw stream
of audio chunks into complete SpeechSegments (start-padded, end-padded).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import numpy as np
import structlog

from server.core.config import VADConfig

logger = structlog.get_logger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    """A complete speech segment extracted by VAD."""

    audio: np.ndarray         # float32, normalized [-1.0, 1.0], mono 16kHz
    sample_rate: int          # always 16000
    start_offset_ms: int      # ms from session start
    end_offset_ms: int
    session_id: str
    duration_ms: int = field(init=False)

    def __post_init__(self) -> None:
        self.duration_ms = self.end_offset_ms - self.start_offset_ms


# ── VAD Engine Protocol ─────────────────────────────────────────────────────

@runtime_checkable
class VADEngine(Protocol):
    """All VAD engines implement this protocol."""

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Return speech probability for this audio chunk (0.0 – 1.0)."""
        ...

    def reset(self) -> None:
        """Reset internal recurrent state."""
        ...


# ── Silero VAD ──────────────────────────────────────────────────────────────

class SileroVAD:
    """Silero VAD v5 via ONNX Runtime.

    Processes 512-sample windows (32ms at 16kHz).
    The ONNX model is stateful — call reset() between sessions.
    Runs on CPU (tiny model; no need to occupy GPU).
    """

    SAMPLE_RATE = 16000
    WINDOW_SIZE = 512   # samples per inference call

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._session = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def load(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # suppress verbose logs

        self._session = ort.InferenceSession(
            self._model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.reset()
        logger.info("Silero VAD loaded", model_path=self._model_path)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Return speech probability for a 512-sample window."""
        if self._session is None:
            raise RuntimeError("SileroVAD not loaded — call load() first")

        x = chunk.astype(np.float32).reshape(1, -1)
        sr = np.array(self.SAMPLE_RATE, dtype=np.int64)

        ort_inputs = {
            "input": x,
            "sr": sr,
            "h": self._h,
            "c": self._c,
        }
        out, self._h, self._c = self._session.run(None, ort_inputs)
        return float(out.squeeze())

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)


# ── WebRTC VAD (fallback) ────────────────────────────────────────────────────

class WebRTCVAD:
    """webrtcvad-based VAD — lightweight CPU fallback.

    Requires: pip install webrtcvad-wheels
    Works with 10/20/30ms frames only at 8/16/32/48kHz.
    """

    FRAME_DURATION_MS = 30   # 480 samples at 16kHz

    def __init__(self, aggressiveness: int = 2) -> None:
        self._aggressiveness = aggressiveness
        self._vad = None

    def load(self) -> None:
        import webrtcvad
        self._vad = webrtcvad.Vad(self._aggressiveness)
        logger.info("WebRTC VAD loaded", aggressiveness=self._aggressiveness)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Return 1.0 if chunk contains speech, 0.0 otherwise."""
        if self._vad is None:
            raise RuntimeError("WebRTCVAD not loaded — call load() first")

        pcm = (chunk * 32767).astype(np.int16).tobytes()
        try:
            is_speech = self._vad.is_speech(pcm, 16000)
        except Exception:
            return 0.0
        return 1.0 if is_speech else 0.0

    def reset(self) -> None:
        pass   # WebRTC VAD is stateless per-chunk


# ── VAD Segmenter State Machine ──────────────────────────────────────────────

class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()
    ENDING = auto()   # grace period: waiting for min_silence_ms to confirm end


class VADSegmenter:
    """Wraps a VADEngine and converts audio chunks into complete SpeechSegments.

    State machine:
      SILENCE  ─(above threshold)──▶  SPEECH
      SPEECH   ─(below threshold)──▶  ENDING
      ENDING   ─(above threshold)──▶  SPEECH   (false alarm, continue)
      ENDING   ─(silence > min_ms)──▶  SILENCE  (emit SpeechSegment)
      SPEECH   ─(duration > max_s)──▶  force-emit SpeechSegment (keep listening)

    Audio is accumulated with speech_pad_ms of padding on both ends.
    """

    def __init__(self, vad: VADEngine, config: VADConfig) -> None:
        self._vad = vad
        self._cfg = config
        self._state = _State.SILENCE
        self._speech_buf: list[np.ndarray] = []
        self._pre_buf: deque[np.ndarray] = deque()   # pre-speech padding buffer
        self._silence_samples = 0
        self._speech_samples = 0
        self._session_start_ms: int = int(time.monotonic() * 1000)
        self._speech_start_ms: int = 0

        pad_samples = int(config.speech_pad_ms / 1000 * 16000)
        pre_buf_size = (pad_samples // config.window_size_samples) + 1
        self._pre_buf_maxlen = pre_buf_size
        self._pre_buf = deque(maxlen=pre_buf_size)

    def process_chunk(
        self,
        chunk: np.ndarray,
        session_id: str,
    ) -> list[SpeechSegment]:
        """Feed one audio chunk (window_size_samples). Returns 0 or 1 SpeechSegments."""
        prob = self._vad.process_chunk(chunk)
        now_ms = int(time.monotonic() * 1000) - self._session_start_ms
        is_speech = prob >= self._cfg.threshold
        min_silence_samples = int(self._cfg.min_silence_duration_ms / 1000 * 16000)
        max_speech_samples = int(self._cfg.max_speech_duration_s * 16000)

        segments: list[SpeechSegment] = []

        if self._state == _State.SILENCE:
            self._pre_buf.append(chunk)
            if is_speech:
                self._state = _State.SPEECH
                self._speech_buf = list(self._pre_buf)
                self._speech_samples = sum(len(c) for c in self._speech_buf)
                self._speech_start_ms = now_ms - self._cfg.speech_pad_ms
                self._silence_samples = 0

        elif self._state == _State.SPEECH:
            self._speech_buf.append(chunk)
            self._speech_samples += len(chunk)

            if not is_speech:
                self._state = _State.ENDING
                self._silence_samples = len(chunk)
            elif self._speech_samples >= max_speech_samples:
                # Force-emit: utterance too long; split here
                seg = self._build_segment(session_id, now_ms)
                if seg is not None:
                    segments.append(seg)
                self._reset_speech()

        elif self._state == _State.ENDING:
            self._speech_buf.append(chunk)
            self._speech_samples += len(chunk)

            if is_speech:
                self._state = _State.SPEECH
                self._silence_samples = 0
            else:
                self._silence_samples += len(chunk)
                if self._silence_samples >= min_silence_samples:
                    # Confirmed end of speech
                    min_speech_samples = int(self._cfg.min_speech_duration_ms / 1000 * 16000)
                    if self._speech_samples >= min_speech_samples:
                        seg = self._build_segment(session_id, now_ms)
                        if seg is not None:
                            segments.append(seg)
                    self._reset_speech()
                    self._state = _State.SILENCE

        return segments

    async def process_chunk_async(
        self,
        chunk: np.ndarray,
        session_id: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> list[SpeechSegment]:
        """Async wrapper — runs inference in the calling thread (VAD is fast enough)."""
        return self.process_chunk(chunk, session_id)

    def _build_segment(self, session_id: str, end_ms: int) -> SpeechSegment | None:
        if not self._speech_buf:
            return None
        audio = np.concatenate(self._speech_buf).astype(np.float32)
        # Clip to valid float range in case of int16 input that wasn't normalized
        audio = np.clip(audio, -1.0, 1.0)
        return SpeechSegment(
            audio=audio,
            sample_rate=16000,
            start_offset_ms=self._speech_start_ms,
            end_offset_ms=end_ms,
            session_id=session_id,
        )

    def _reset_speech(self) -> None:
        self._speech_buf = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._pre_buf.clear()
        self._vad.reset()

    def reset(self) -> None:
        """Full reset — call between independent audio sessions."""
        self._reset_speech()
        self._state = _State.SILENCE
        self._session_start_ms = int(time.monotonic() * 1000)

    @property
    def is_in_speech(self) -> bool:
        return self._state in (_State.SPEECH, _State.ENDING)


def create_vad(config: VADConfig) -> tuple[VADEngine, VADSegmenter]:
    """Factory — returns a loaded VAD engine + segmenter."""
    if config.engine == "silero":
        engine: VADEngine = SileroVAD(config.model_path)
    else:
        engine = WebRTCVAD()

    engine.load()  # type: ignore[attr-defined]
    segmenter = VADSegmenter(engine, config)
    return engine, segmenter
