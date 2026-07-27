"""
Configuration schema for the Nepali speech translator server.

All settings are loaded from a YAML file and validated by Pydantic v2.
Every module receives a typed Config object — no raw dict access anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class TLSConfig(BaseModel):
    enabled: bool = False
    cert_file: str = "/etc/translator/cert.pem"
    key_file: str = "/etc/translator/key.pem"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    tls: TLSConfig = Field(default_factory=TLSConfig)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_concurrent_sessions: int = 4
    # Opus is NOT implemented -- no encode/decode path exists. Restricted to
    # "pcm" so a config asking for opus fails validation instead of silently
    # continuing to send raw PCM.
    audio_codec: Literal["pcm"] = "pcm"


class LanguageConfig(BaseModel):
    source: str = "npi"   # ISO 639-3
    target: str = "eng"   # ISO 639-3


class SystemConfig(BaseModel):
    cuda_device: int = 0
    cuda_memory_fraction: float = 0.85
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"


class VADConfig(BaseModel):
    engine: Literal["silero", "webrtc"] = "silero"
    model_path: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: int = 30
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200
    window_size_samples: int = 512


class NoiseReductionConfig(BaseModel):
    # Off by default: DeepFilterNet needs a Rust toolchain to build and is
    # incompatible with torchaudio 2.x, so it silently fell back to passthrough
    # while logging a warning on every start.
    enabled: bool = False
    engine: Literal["deepfilter", "rnnoise"] = "deepfilter"
    model_path: str = "models/DeepFilterNet3"


class ASRConfig(BaseModel):
    engine: Literal["faster_whisper", "whisper_cpp"] = "faster_whisper"
    model: str = "large-v3"
    model_path: str = "models/faster-whisper-large-v3"
    compute_type: Literal["int8", "float16", "float32"] = "int8"
    device: Literal["cuda", "cpu"] = "cuda"
    language: str = "ne"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: float = 0.0
    word_timestamps: bool = True

    # Feeding the previous (possibly wrong) transcript back into the decoder
    # propagates errors forward across utterances. Whisper is especially prone
    # to this on lower-resource languages, where one bad segment can steer
    # every subsequent one. Off by default.
    condition_on_previous_text: bool = False
    initial_prompt: str = ""
    partial_interval_s: float = 1.0
    stability_window: int = 3

    # Temperature fallback. Whisper's quality guards work by *retrying* a failed
    # decode at successively higher temperature; with a single scalar there is
    # no fallback, so a greedy decode that falls into a repetition loop
    # ("प्रतान प्रतान प्रतान ...") stays stuck there.
    temperature_fallback: list[float] = Field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    # Detects degenerate/repetitive output and triggers the fallback above.
    compression_ratio_threshold: float = 2.4
    # Treat a segment as failed when average log-probability falls below this.
    log_prob_threshold: float = -1.0
    # Suppress transcription of segments the model considers non-speech.
    no_speech_threshold: float = 0.6
    repetition_penalty: float = 1.1
    # Greedy decoding for mid-utterance partials: they are re-transcribed every
    # partial_interval_s and thrown away, so beam search there is wasted GPU.
    partial_beam_size: int = 1

    @model_validator(mode="after")
    def validate_device(self) -> ASRConfig:
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        return self


class TranslationConfig(BaseModel):
    engine: Literal["seamless", "nllb", "marian"] = "seamless"
    model: str = "facebook/seamless-m4t-v2-large"
    model_path: str = "models/seamless-m4t-v2-large"
    device: Literal["cuda", "cpu"] = "cuda"
    torch_dtype: Literal["float16", "float32"] = "float16"
    source_lang: str = "npi"
    target_lang: str = "eng"
    max_new_tokens: int = 256
    num_beams: int = 4
    context_window_chars: int = 200
    lru_cache_size: int = 1000

    @model_validator(mode="after")
    def validate_device(self) -> TranslationConfig:
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
                    self.torch_dtype = "float32"
            except ImportError:
                self.device = "cpu"
                self.torch_dtype = "float32"
        return self


class TTSConfig(BaseModel):
    engine: Literal["piper", "kokoro"] = "piper"
    model_path: str = "models/piper/en_US-lessac-medium.onnx"
    config_path: str = "models/piper/en_US-lessac-medium.onnx.json"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sentence_silence_s: float = 0.15
    use_gpu: bool = False


class QueueSizeConfig(BaseModel):
    audio_chunk_maxsize: int = 10
    speech_segment_maxsize: int = 5
    tts_audio_maxsize: int = 20


class IncrementalConfig(BaseModel):
    enabled: bool = True
    min_commit_words: int = 1
    max_buffer_chars: int = 80


class PipelineConfig(BaseModel):
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)
    queues: QueueSizeConfig = Field(default_factory=QueueSizeConfig)


class PrometheusConfig(BaseModel):
    enabled: bool = True
    port: int = 9090


class MonitoringConfig(BaseModel):
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    latency_window_size: int = 100


class Config(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    noise_reduction: NoiseReductionConfig = Field(default_factory=NoiseReductionConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> Config:
        """Return a Config with all defaults — useful for testing."""
        return cls()
