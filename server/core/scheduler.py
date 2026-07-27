"""
Component registry, startup sequencer, and OOM recovery scheduler.

ModelRegistry holds the loaded ASR, translation, and TTS engines.
Startup is sequential (one model at a time) to avoid GPU memory spikes.
OOM recovery is handled by releasing models in reverse priority order.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog

from server.asr.engine import ASREngine
from server.audio.noise import NoiseReducer, create_noise_reducer
from server.audio.vad import VADEngine, VADSegmenter, create_vad
from server.core.config import Config
from server.core.gpu_manager import GPUMemoryManager
from server.translation.engine import TranslationEngine
from server.tts.engine import TTSEngine

logger = structlog.get_logger(__name__)


@dataclass
class ComponentBundle:
    """All loaded pipeline components, passed to AsyncPipeline."""

    vad_engine: VADEngine
    vad_segmenter: VADSegmenter
    noise_reducer: NoiseReducer
    asr: ASREngine
    translator: TranslationEngine
    tts: TTSEngine
    gpu_manager: GPUMemoryManager


class ModelRegistry:
    """Loads and tracks all pipeline models in correct order.

    Load order: VAD → Whisper → SeamlessM4T → Piper.
    Each model is warmed up before the next one loads.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._gpu = GPUMemoryManager(
            device_index=config.system.cuda_device,
            safety_margin_gb=1.0,
        )
        self._bundle: Optional[ComponentBundle] = None

    async def load_all(self) -> ComponentBundle:
        """Load all models sequentially. Returns a ComponentBundle."""
        logger.info("Starting model load sequence")
        t_start = time.perf_counter()

        # 1. VAD (CPU — fast, load first)
        logger.info("Loading VAD", engine=self._config.vad.engine)
        vad_engine, vad_segmenter = await asyncio.get_running_loop().run_in_executor(
            None, create_vad, self._config.vad
        )

        # 2. Noise reducer (optional)
        logger.info("Loading noise reducer", engine=self._config.noise_reduction.engine)
        noise_reducer = await asyncio.get_running_loop().run_in_executor(
            None, create_noise_reducer, self._config.noise_reduction
        )

        # 3. ASR (GPU — largest allocation first so we fail fast on low VRAM)
        asr = self._create_asr()
        logger.info("Loading ASR", engine=self._config.asr.engine, model=self._config.asr.model)
        await asyncio.get_running_loop().run_in_executor(None, asr.load)
        self._gpu.register_loaded(f"faster_whisper_{self._config.asr.model.replace('-', '_')}_{self._config.asr.compute_type}")

        # 4. Translation
        translator = self._create_translator()
        logger.info("Loading translation", engine=self._config.translation.engine)
        await asyncio.get_running_loop().run_in_executor(None, translator.load)
        self._gpu.register_loaded("seamless_m4t_v2_large_fp16")

        # 5. TTS (CPU)
        tts = self._create_tts()
        logger.info("Loading TTS", engine=self._config.tts.engine)
        await asyncio.get_running_loop().run_in_executor(None, tts.load)

        self._gpu.log_status()
        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info("All models loaded", total_load_ms=round(elapsed))

        self._bundle = ComponentBundle(
            vad_engine=vad_engine,
            vad_segmenter=vad_segmenter,
            noise_reducer=noise_reducer,
            asr=asr,
            translator=translator,
            tts=tts,
            gpu_manager=self._gpu,
        )
        return self._bundle

    def _create_asr(self) -> ASREngine:
        from server.asr.whisper import FasterWhisperASR
        return FasterWhisperASR(self._config.asr)

    def _create_translator(self) -> TranslationEngine:
        if self._config.translation.engine == "seamless":
            from server.translation.seamless import SeamlessM4TTranslator
            return SeamlessM4TTranslator(self._config.translation)
        elif self._config.translation.engine == "nllb":
            from server.translation.nllb import NLLBTranslator
            return NLLBTranslator(self._config.translation)
        raise ValueError(f"Unknown translation engine: {self._config.translation.engine}")

    def _create_tts(self) -> TTSEngine:
        if self._config.tts.engine == "piper":
            from server.tts.piper import PiperTTS
            return PiperTTS(self._config.tts)
        elif self._config.tts.engine == "kokoro":
            from server.tts.kokoro import KokoroTTS
            return KokoroTTS()
        raise ValueError(f"Unknown TTS engine: {self._config.tts.engine}")

    async def unload_all(self) -> None:
        """Gracefully release all GPU resources."""
        if self._bundle is None:
            return

        # Reverse priority: TTS → translation → ASR
        try:
            self._bundle.tts.unload()
        except Exception:
            pass
        try:
            self._bundle.translator.unload()
            self._gpu.register_unloaded("seamless_m4t_v2_large_fp16")
        except Exception:
            pass
        try:
            self._bundle.asr.unload()
        except Exception:
            pass

        self._bundle = None
        logger.info("All models unloaded")

    @property
    def bundle(self) -> Optional[ComponentBundle]:
        return self._bundle

    @property
    def gpu_manager(self) -> GPUMemoryManager:
        return self._gpu
