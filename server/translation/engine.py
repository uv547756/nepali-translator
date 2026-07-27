"""
Abstract translation engine interface and shared data types.

All translation backends implement TranslationEngine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Output from one translation inference call."""

    source_text: str
    translated_text: str
    source_lang: str       # ISO 639-3: npi, eng, hin
    target_lang: str
    latency_ms: float
    from_cache: bool = False


class TranslationEngine(ABC):
    """Abstract base for all translation backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory. Called once at startup."""

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
    ) -> TranslationResult:
        """Translate text from source_lang to target_lang.

        context: The previously translated text — passed to the model as a
                 soft coherence hint so successive chunks use consistent vocabulary.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU/CPU memory."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def vram_usage_gb(self) -> float: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...
