"""Unit tests for the configuration schema."""

import pytest
from server.core.config import Config, ASRConfig, TranslationConfig


def test_default_config_valid():
    cfg = Config.default()
    assert cfg.system.cuda_device == 0
    assert cfg.asr.engine == "faster_whisper"
    assert cfg.translation.engine == "seamless"
    assert cfg.tts.engine == "piper"


def test_config_from_yaml(tmp_path):
    yaml_content = """
system:
  cuda_device: 1
  log_level: DEBUG
asr:
  beam_size: 3
  language: hi
"""
    f = tmp_path / "test.yaml"
    f.write_text(yaml_content)
    cfg = Config.from_yaml(str(f))
    assert cfg.system.cuda_device == 1
    assert cfg.system.log_level == "DEBUG"
    assert cfg.asr.beam_size == 3
    assert cfg.asr.language == "hi"
    # Unspecified fields have defaults
    assert cfg.tts.engine == "piper"


def test_invalid_log_level_rejected():
    with pytest.raises(Exception):
        Config.model_validate({"system": {"log_level": "VERBOSE"}})


def test_language_defaults():
    cfg = Config.default()
    assert cfg.system.language.source == "npi"
    assert cfg.system.language.target == "eng"
    assert cfg.translation.source_lang == "npi"
    assert cfg.translation.target_lang == "eng"
