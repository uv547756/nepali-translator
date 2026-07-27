"""
Kokoro TTS backend — fallback to Piper.

Kokoro is a high-quality ONNX-based TTS model. Use it when Piper voices
are unavailable or when higher synthesis quality is required at the cost
of ~2× latency (still well within the 200ms budget for short sentences).

Requires: pip install kokoro-onnx
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import numpy as np
import structlog

from server.tts.engine import TTSEngine, SynthesisResult
from server.tts.piper import _split_at_boundary

logger = structlog.get_logger(__name__)


class KokoroTTS(TTSEngine):
    """Kokoro ONNX TTS — higher quality English synthesis than Piper."""

    DEFAULT_SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: str = "models/kokoro/kokoro-v0_19.onnx",
        voices_path: str = "models/kokoro/voices.bin",
        voice: str = "af",
        speed: float = 1.0,
    ) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self._voice = voice
        self._speed = speed
        self._kokoro = None
        self._sr = self.DEFAULT_SAMPLE_RATE
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")
        self._loaded = False

    def load(self) -> None:
        try:
            from kokoro_onnx import Kokoro  # type: ignore[import]

            t0 = time.perf_counter()
            self._kokoro = Kokoro(self._model_path, self._voices_path)
            self._sr = self.DEFAULT_SAMPLE_RATE

            # Warm-up
            self._synthesize_sync("Hello.")
            self._loaded = True
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Kokoro TTS loaded", model=self._model_path, load_ms=round(elapsed))
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run: pip install kokoro-onnx"
            ) from exc

    def unload(self) -> None:
        self._kokoro = None
        self._loaded = False

    async def synthesize(self, text: str) -> SynthesisResult:
        if not self._loaded or self._kokoro is None:
            raise RuntimeError("KokoroTTS not loaded — call load() first")

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
        buffer = ""
        async for chunk in text_iter:
            if not chunk:
                continue
            buffer += (" " if buffer else "") + chunk
            complete, buffer = _split_at_boundary(buffer)
            for sentence in complete:
                sentence = sentence.strip()
                if sentence:
                    result = await self.synthesize(sentence)
                    int16 = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)
                    await audio_queue.put(int16.tobytes())

        remainder = buffer.strip()
        if remainder:
            result = await self.synthesize(remainder)
            int16 = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)
            await audio_queue.put(int16.tobytes())

        await audio_queue.put(b"")

    def _synthesize_sync(self, text: str) -> np.ndarray:
        assert self._kokoro is not None
        samples, sr = self._kokoro.create(text, voice=self._voice, speed=self._speed, lang="en-us")
        return samples.astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def model_name(self) -> str:
        return "kokoro-v0_19"
