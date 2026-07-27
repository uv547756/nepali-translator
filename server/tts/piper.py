"""
Piper TTS backend — primary TTS engine.

Piper uses ONNX Runtime for inference and espeak-ng for phonemization.
It synthesizes English (and other languages) at ~40-80ms per short sentence
on CPU, well within the 200ms latency budget without touching the GPU.

Sentence streaming: text is accumulated in a buffer, split at sentence
boundaries, and each complete sentence is synthesized and pushed to the
audio queue immediately — without waiting for the full utterance.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import structlog

from server.core.config import TTSConfig
from server.tts.engine import TTSEngine, SynthesisResult

logger = structlog.get_logger(__name__)

# Sentence boundary patterns
_HARD_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')
_SOFT_BOUNDARY = re.compile(r'(?<=[,;:])\s+')

# Minimum chars before attempting a soft boundary split
_SOFT_SPLIT_THRESHOLD = 60


def _split_at_boundary(text: str) -> tuple[list[str], str]:
    """Split text at sentence boundaries.

    Returns (list_of_complete_sentences, incomplete_remainder).
    Hard boundaries ([.!?] followed by capital) always split.
    Soft boundaries ([,;:]) split only when text exceeds threshold.
    """
    parts = _HARD_BOUNDARY.split(text)
    if len(parts) > 1:
        return parts[:-1], parts[-1]

    if len(text) >= _SOFT_SPLIT_THRESHOLD:
        parts = _SOFT_BOUNDARY.split(text, maxsplit=1)
        if len(parts) > 1:
            return [parts[0]], parts[1]

    return [], text


class PiperTTS(TTSEngine):
    """Piper ONNX TTS with sentence-level streaming synthesis.

    Requires:
      - piper-tts Python package (or piper-phonemize + onnxruntime)
      - espeak-ng system library (for phonemization)
      - A .onnx model file and its matching .onnx.json config
    """

    def __init__(self, config: TTSConfig) -> None:
        self._config = config
        self._voice = None
        self._sr: int = 22050
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper")
        self._loaded = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def load(self) -> None:
        from piper import PiperVoice  # type: ignore[import]

        t0 = time.perf_counter()
        self._voice = PiperVoice.load(
            self._config.model_path,
            config_path=self._config.config_path,
            use_cuda=self._config.use_gpu,
        )

        # Read sample rate from the model config
        cfg_path = Path(self._config.config_path)
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg_data = json.load(f)
            self._sr = cfg_data.get("audio", {}).get("sample_rate", 22050)

        # Warm-up
        self._synthesize_sync("Hello.")

        self._loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Piper TTS loaded",
            model=self._config.model_path,
            sample_rate=self._sr,
            load_ms=round(elapsed),
        )

    def unload(self) -> None:
        self._voice = None
        self._loaded = False

    # ── Synthesis ───────────────────────────────────────────────────────────

    async def synthesize(self, text: str) -> SynthesisResult:
        if not self._loaded or self._voice is None:
            raise RuntimeError("PiperTTS not loaded — call load() first")

        text = text.strip()
        if not text:
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self._sr,
                text=text,
                latency_ms=0.0,
                duration_s=0.0,
            )

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        audio_f32 = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return SynthesisResult(
            audio=audio_f32,
            sample_rate=self._sr,
            text=text,
            latency_ms=latency_ms,
            duration_s=len(audio_f32) / self._sr,
        )

    async def synthesize_streaming(
        self,
        text_iter: AsyncIterator[str],
        audio_queue: asyncio.Queue[bytes],
    ) -> None:
        """Consume text chunks, push synthesized PCM to audio_queue at sentence boundaries."""
        buffer = ""

        async for chunk in text_iter:
            if not chunk:
                continue
            buffer += (" " if buffer else "") + chunk
            complete, buffer = _split_at_boundary(buffer)

            for sentence in complete:
                sentence = sentence.strip()
                if sentence:
                    pcm = await self._synthesize_to_bytes(sentence)
                    await audio_queue.put(pcm)

        # Flush remainder
        remainder = buffer.strip()
        if remainder:
            pcm = await self._synthesize_to_bytes(remainder)
            await audio_queue.put(pcm)

        await audio_queue.put(b"")   # sentinel: end of utterance

    async def _synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize and return Int16 LE PCM bytes."""
        loop = asyncio.get_running_loop()
        audio_f32 = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
        )
        # Add inter-sentence silence
        silence_samples = int(self._config.sentence_silence_s * self._sr)
        if silence_samples > 0:
            audio_f32 = np.concatenate([
                audio_f32,
                np.zeros(silence_samples, dtype=np.float32),
            ])
        int16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        return int16.tobytes()

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis using Piper — runs inside the executor."""
        assert self._voice is not None

        synthesize_args = {
            "length_scale": self._config.length_scale,
            "noise_scale": self._config.noise_scale,
            "noise_w": self._config.noise_w,
        }
        if hasattr(self._voice, "speaker_id"):
            synthesize_args["speaker_id"] = self._config.speaker_id

        audio_chunks: list[bytes] = []
        for audio_bytes in self._voice.synthesize_stream_raw(text, **synthesize_args):
            audio_chunks.append(audio_bytes)

        raw = b"".join(audio_chunks)
        int16 = np.frombuffer(raw, dtype=np.int16)
        return int16.astype(np.float32) / 32768.0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def model_name(self) -> str:
        return f"piper-{Path(self._config.model_path).stem}"
