"""
Noise reduction adapters.

Primary: DeepFilterNet3 — GPU-accelerated neural noise suppression.
Fallback: RNNoise — lightweight recurrent model, CPU-only.

Both adapters expose the same interface: process(audio) → cleaned_audio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import structlog

from server.core.config import NoiseReductionConfig

logger = structlog.get_logger(__name__)


class NoiseReducer(ABC):
    """Abstract base for noise reduction backends."""

    @abstractmethod
    def load(self) -> None:
        """Initialize and warm up the model."""

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Remove noise from a float32 audio array. Returns float32 same shape."""

    @abstractmethod
    def unload(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...


class DeepFilterNetReducer(NoiseReducer):
    """DeepFilterNet3 via the deepfilternet Python package.

    Processes arbitrary-length float32 audio at 48kHz internally
    (the library resamples from/to 16kHz automatically via a convenience wrapper).
    """

    def __init__(self, model_path: str = "models/DeepFilterNet3") -> None:
        self._model_path = model_path
        self._model = None
        self._df_state = None
        self._loaded = False

    def load(self) -> None:
        try:
            from df import init_df, enhance
            self._enhance = enhance
            self._model, self._df_state, _ = init_df(self._model_path)
            self._loaded = True
            logger.info("DeepFilterNet loaded", model_path=self._model_path)
        except Exception as exc:
            logger.warning(
                "DeepFilterNet failed to load — noise reduction disabled",
                error=str(exc),
            )
            self._loaded = False

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if not self._loaded or self._model is None:
            return audio

        import torch
        # deepfilternet expects float32 tensor shape (1, samples)
        tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        enhanced = self._enhance(self._model, self._df_state, tensor)
        return enhanced.squeeze(0).numpy()

    def unload(self) -> None:
        self._model = None
        self._df_state = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class RNNoiseReducer(NoiseReducer):
    """RNNoise via the rnnoise-python package (CPU-only, very fast).

    Processes 480-sample chunks at 48kHz. Input is resampled automatically.
    """

    def __init__(self) -> None:
        self._denoiser = None
        self._loaded = False

    def load(self) -> None:
        try:
            import rnnoise
            self._denoiser = rnnoise.RNNoise()
            self._loaded = True
            logger.info("RNNoise loaded")
        except Exception as exc:
            logger.warning("RNNoise failed to load — noise reduction disabled", error=str(exc))
            self._loaded = False

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if not self._loaded or self._denoiser is None:
            return audio

        # RNNoise works on 48kHz; resample if needed
        import scipy.signal as ss
        if sample_rate != 48000:
            audio_48k = ss.resample_poly(audio, 48000, sample_rate).astype(np.float32)
        else:
            audio_48k = audio

        CHUNK = 480
        out_chunks: list[np.ndarray] = []
        for i in range(0, len(audio_48k), CHUNK):
            chunk = audio_48k[i : i + CHUNK]
            if len(chunk) < CHUNK:
                chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
            vad_prob, denoised = self._denoiser.process_frame(chunk)
            out_chunks.append(denoised)

        out_48k = np.concatenate(out_chunks)[: len(audio_48k)]

        if sample_rate != 48000:
            return ss.resample_poly(out_48k, sample_rate, 48000).astype(np.float32)
        return out_48k

    def unload(self) -> None:
        self._denoiser = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class PassthroughReducer(NoiseReducer):
    """No-op reducer used when noise reduction is disabled."""

    def load(self) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        return audio

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True


def create_noise_reducer(config: NoiseReductionConfig) -> NoiseReducer:
    """Factory — returns an initialized noise reducer per config."""
    if not config.enabled:
        return PassthroughReducer()

    if config.engine == "deepfilter":
        reducer: NoiseReducer = DeepFilterNetReducer(config.model_path)
    else:
        reducer = RNNoiseReducer()

    reducer.load()
    return reducer
