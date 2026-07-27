"""
SeamlessM4T v2 translation backend — text-to-text mode.

Uses SeamlessM4Tv2ForTextToText (not the S2T variant) so that ASR and
translation are independently controllable. Loads in FP16 on CUDA.

LRU cache: repeated short phrases (greetings, common words) are served
instantly from cache without GPU inference.

Language codes use ISO 639-3 as required by the SeamlessM4T tokenizer:
  Nepali → npi
  English → eng
  Hindi   → hin
"""

from __future__ import annotations

import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import structlog

from server.core.config import TranslationConfig
from server.translation.engine import TranslationEngine, TranslationResult

logger = structlog.get_logger(__name__)


class SeamlessM4TTranslator(TranslationEngine):
    """SeamlessM4T v2 text-to-text translation, FP16, CUDA.

    All inference runs in a single-threaded ThreadPoolExecutor so the
    PyTorch CUDA context stays on one OS thread.
    """

    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self._model = None
        self._processor = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seamless")
        self._loaded = False
        # LRU cache keyed on (normalized_text, target_lang)
        self._cache: dict[tuple[str, str], str] = {}
        self._cache_order: list[tuple[str, str]] = []

    # ── Lifecycle ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_model_path(local_path: str) -> str:
        """Return local_path if it contains weight files, else fall back to HF hub ID."""
        from pathlib import Path
        p = Path(local_path)
        if p.is_dir():
            has_weights = (
                (p / "model.safetensors").exists()
                or (p / "pytorch_model.bin").exists()
                or bool(list(p.glob("model-*.safetensors")))
                or bool(list(p.glob("pytorch_model-*.bin")))
            )
            if has_weights:
                return local_path
        return "facebook/seamless-m4t-v2-large"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

        t0 = time.perf_counter()
        dtype = torch.float16 if self._config.torch_dtype == "float16" else torch.float32

        source = self._resolve_model_path(self._config.model_path)
        if source != self._config.model_path:
            logger.warning(
                "Local model path has no weights — loading from HuggingFace hub",
                local_path=self._config.model_path,
                hub_id=source,
            )

        self._processor = AutoProcessor.from_pretrained(source)
        self._model = SeamlessM4Tv2ForTextToText.from_pretrained(
            source,
            torch_dtype=dtype,
        )

        if self._config.device == "cuda":
            self._model = self._model.cuda()

        self._model.eval()

        # Warm-up
        self._translate_sync("hello", "eng", "eng")

        self._loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "SeamlessM4T v2 loaded",
            model_path=self._config.model_path,
            dtype=self._config.torch_dtype,
            device=self._config.device,
            load_ms=round(elapsed),
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("SeamlessM4T unloaded")

    # ── Inference ───────────────────────────────────────────────────────────

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
    ) -> TranslationResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("SeamlessM4TTranslator not loaded — call load() first")

        text = text.strip()
        if not text:
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=0.0,
            )

        cache_key = (text.lower(), target_lang)
        if cache_key in self._cache:
            return TranslationResult(
                source_text=text,
                translated_text=self._cache[cache_key],
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=0.0,
                from_cache=True,
            )

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        translated = await loop.run_in_executor(
            self._executor,
            self._translate_sync,
            text,
            source_lang,
            target_lang,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        self._cache_put(cache_key, translated)

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=latency_ms,
        )

    def _translate_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Synchronous inference — runs inside the executor thread."""
        import torch

        assert self._processor is not None and self._model is not None

        inputs = self._processor(
            text=text,
            src_lang=source_lang,
            return_tensors="pt",
        )

        if self._config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                tgt_lang=target_lang,
                num_beams=self._config.num_beams,
                max_new_tokens=self._config.max_new_tokens,
            )

        translated = self._processor.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
        )
        return translated.strip()

    # ── LRU Cache ───────────────────────────────────────────────────────────

    def _cache_put(self, key: tuple[str, str], value: str) -> None:
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self._config.lru_cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._cache_order.append(key)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_gb(self) -> float:
        # SeamlessM4T v2 large in FP16 ≈ 4.5 GB
        return 4.5 if self._loaded else 0.0

    @property
    def model_name(self) -> str:
        return "seamless-m4t-v2-large"
