"""
Abstract TTS engine interface and shared data types.

All TTS backends implement TTSEngine. The synthesize_streaming() method
is the primary path for real-time output: it consumes an async text iterator
and pushes synthesized PCM bytes to an asyncio.Queue without waiting for the
entire utterance to complete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import asyncio
import numpy as np


@dataclass
class SynthesisResult:
    """Output from one TTS synthesis call."""

    audio: np.ndarray     # float32, normalized [-1, 1]
    sample_rate: int      # Piper default: 22050
    text: str
    latency_ms: float
    duration_s: float


class TTSEngine(ABC):
    """Abstract base for all TTS backends."""

    @abstractmethod
    def load(self) -> None:
        """Initialize model and warm up. Called once at startup."""

    @abstractmethod
    async def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize text to audio. Awaits full synthesis before returning."""

    @abstractmethod
    async def synthesize_streaming(
        self,
        text_iter: AsyncIterator[str],
        audio_queue: asyncio.Queue[bytes],
    ) -> None:
        """Consume text chunks, synthesize at sentence boundaries.

        Pushes Int16 LE PCM bytes to audio_queue as each sentence completes,
        without waiting for the full utterance. Sends None sentinel when done.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...
