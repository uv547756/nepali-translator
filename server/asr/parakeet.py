"""
NVIDIA Parakeet ASR stub.

Parakeet (via NVIDIA NeMo) is English-only. It is included here as a
placeholder so the engine registry can enumerate it; any attempt to use it
for Nepali raises NotImplementedError.

To enable Parakeet for English transcription tasks, install NeMo:
  pip install nemo_toolkit[asr]
"""

from __future__ import annotations

import numpy as np

from server.asr.engine import ASREngine, ASRResult
from server.core.config import ASRConfig


class ParakeetASR(ASREngine):
    """NVIDIA Parakeet-TDT — English-only ASR engine (NeMo-based)."""

    _SUPPORTED_LANGUAGES = {"en"}

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model = None
        self._loaded = False

    def load(self) -> None:
        if self._config.language not in self._SUPPORTED_LANGUAGES:
            raise NotImplementedError(
                f"Parakeet does not support language '{self._config.language}'. "
                "Use faster_whisper for Nepali (ne) transcription."
            )

        try:
            import nemo.collections.asr as nemo_asr  # type: ignore[import]

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-1.1b"
            )
            if self._config.device == "cuda":
                self._model = self._model.cuda()
            self._model.eval()
            self._loaded = True
        except ImportError as exc:
            raise RuntimeError(
                "NeMo toolkit is not installed. "
                "Run: pip install nemo_toolkit[asr]"
            ) from exc

    async def transcribe(
        self,
        audio: np.ndarray,
        session_id: str = "",
        previous_text: str = "",
        is_partial: bool = False,
    ) -> ASRResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("ParakeetASR not loaded — call load() first")
        if self._config.language not in self._SUPPORTED_LANGUAGES:
            raise NotImplementedError(
                f"Parakeet does not support language '{self._config.language}'"
            )
        raise NotImplementedError("Parakeet transcription not yet wired for streaming use")

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_gb(self) -> float:
        return 4.2 if self._loaded else 0.0

    @property
    def model_name(self) -> str:
        return "nvidia-parakeet-tdt-1.1b"
