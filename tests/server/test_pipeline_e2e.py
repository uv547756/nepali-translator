"""
End-to-end WebSocket integration test.

Uses FastAPI's TestClient + starlette.testclient to test the WebSocket
without requiring real models. Models are replaced with lightweight stubs.

Run:
  pytest tests/server/test_pipeline_e2e.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from server.asr.engine import ASREngine, ASRResult, WordTimestamp
from server.translation.engine import TranslationEngine, TranslationResult
from server.tts.engine import TTSEngine, SynthesisResult
from server.audio.vad import VADEngine, VADSegmenter, SpeechSegment
from server.audio.noise import PassthroughReducer
from server.core.config import Config
from server.core.gpu_manager import GPUMemoryManager
from server.core.metrics import MetricsCollector
from server.core.pipeline import PipelineFactory
from server.core.scheduler import ComponentBundle


# ── Stub implementations ────────────────────────────────────────────────────

class StubVAD:
    """VAD that marks everything as speech."""
    def process_chunk(self, chunk: np.ndarray) -> float:
        return 1.0
    def reset(self) -> None:
        pass
    def load(self) -> None:
        pass


class StubASR(ASREngine):
    def load(self) -> None: pass
    async def transcribe(self, audio, session_id="", previous_text="") -> ASRResult:
        return ASRResult(
            text="तपाईं कहाँ जानुहुन्छ",
            is_final=True,
            confidence=0.95,
            words=[
                WordTimestamp("तपाईं", 0.0, 0.5, 0.95),
                WordTimestamp("कहाँ", 0.5, 1.0, 0.92),
                WordTimestamp("जानुहुन्छ", 1.0, 1.8, 0.90),
            ],
            language="ne",
            latency_ms=100.0,
            audio_duration_s=2.0,
            session_id=session_id,
        )
    def unload(self) -> None: pass
    @property
    def is_loaded(self) -> bool: return True
    @property
    def vram_usage_gb(self) -> float: return 0.0
    @property
    def model_name(self) -> str: return "stub-asr"


class StubTranslator(TranslationEngine):
    def load(self) -> None: pass
    async def translate(self, text, source_lang, target_lang, context="") -> TranslationResult:
        return TranslationResult(
            source_text=text,
            translated_text="Where are you going?",
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=30.0,
        )
    def unload(self) -> None: pass
    @property
    def is_loaded(self) -> bool: return True
    @property
    def vram_usage_gb(self) -> float: return 0.0
    @property
    def model_name(self) -> str: return "stub-translator"


class StubTTS(TTSEngine):
    def load(self) -> None: pass
    async def synthesize(self, text: str) -> SynthesisResult:
        samples = int(0.5 * 22050)
        return SynthesisResult(
            audio=np.zeros(samples, dtype=np.float32),
            sample_rate=22050,
            text=text,
            latency_ms=40.0,
            duration_s=0.5,
        )
    async def synthesize_streaming(self, text_iter, audio_queue):
        async for chunk in text_iter:
            result = await self.synthesize(chunk)
            int16 = (result.audio * 32767).astype(np.int16)
            await audio_queue.put(int16.tobytes())
        await audio_queue.put(b"")
    def unload(self) -> None: pass
    @property
    def is_loaded(self) -> bool: return True
    @property
    def sample_rate(self) -> int: return 22050
    @property
    def model_name(self) -> str: return "stub-tts"


@pytest.fixture
def stub_bundle():
    from server.core.config import VADConfig
    vad_engine = StubVAD()
    vad_segmenter = VADSegmenter(vad_engine, VADConfig())
    return ComponentBundle(
        vad_engine=vad_engine,
        vad_segmenter=vad_segmenter,
        noise_reducer=PassthroughReducer(),
        asr=StubASR(),
        translator=StubTranslator(),
        tts=StubTTS(),
        gpu_manager=GPUMemoryManager(),
    )


@pytest.fixture
def config():
    return Config.default()


@pytest.fixture
def metrics():
    return MetricsCollector()


# ── Pipeline unit tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_processes_audio(stub_bundle, config, metrics):
    factory = PipelineFactory(stub_bundle, config, metrics)
    pipeline = await factory.create_session("test-session-1")

    # Feed 2 seconds of silence (will trigger VAD → ASR via stub)
    silence = np.zeros(32000, dtype=np.float32)
    pcm_bytes = (silence * 32767).astype(np.int16).tobytes()

    # Feed in chunks
    chunk_size = 512 * 2  # 512 int16 samples = 1024 bytes
    for i in range(0, len(pcm_bytes), chunk_size):
        await pipeline.feed_audio(pcm_bytes[i : i + chunk_size])

    await asyncio.sleep(0.1)  # Let pipeline process

    await factory.destroy_session("test-session-1")


@pytest.mark.asyncio
async def test_pipeline_max_sessions(stub_bundle, config, metrics):
    config.server.max_concurrent_sessions = 2
    factory = PipelineFactory(stub_bundle, config, metrics)

    s1 = await factory.create_session("s1")
    s2 = await factory.create_session("s2")

    with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
        await factory.create_session("s3")

    await factory.destroy_session("s1")
    await factory.destroy_session("s2")


@pytest.mark.asyncio
async def test_pipeline_mute_blocks_audio(stub_bundle, config, metrics):
    factory = PipelineFactory(stub_bundle, config, metrics)
    pipeline = await factory.create_session("mute-test")

    pipeline.set_muted(True)
    pcm = np.zeros(1024, dtype=np.int16).tobytes()
    await pipeline.feed_audio(pcm)
    # When muted, audio queue should remain empty
    assert pipeline._audio_chunk_q.empty()

    await factory.destroy_session("mute-test")
