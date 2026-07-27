"""
Abstract ASR engine interface and shared data types.

All ASR backends implement ASREngine. Callers use only this interface,
enabling runtime model swapping without changing any downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable

import numpy as np


@dataclass
class WordTimestamp:
    """A single word with timing and confidence from ASR."""

    word: str
    start: float      # seconds from segment start
    end: float
    probability: float


@dataclass
class ASRResult:
    """Output from one ASR inference call."""

    text: str
    is_final: bool
    confidence: float          # mean word probability
    words: list[WordTimestamp]
    language: str              # BCP-47 detected language
    latency_ms: float
    audio_duration_s: float
    session_id: str = ""
    is_partial: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_partial = not self.is_final

    @property
    def word_list(self) -> list[str]:
        """Plain word strings, stripped of leading/trailing whitespace."""
        return [w.word.strip() for w in self.words if w.word.strip()]


class ASREngine(ABC):
    """Abstract base for all ASR backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once at startup."""

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        session_id: str = "",
        previous_text: str = "",
        is_partial: bool = False,
    ) -> ASRResult:
        """Transcribe a complete speech segment.

        audio: float32, normalized [-1, 1], mono, 16kHz.
        previous_text: prior transcript to condition the model for context continuity.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU/CPU memory. Model must be re-loaded before next use."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def vram_usage_gb(self) -> float:
        """Estimated current VRAM usage in gigabytes."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging and metrics."""
