"""
NLLB-200 distilled-1.3B translation backend — fallback to SeamlessM4T.

Faster and lighter than SeamlessM4T v2 large; useful when VRAM is constrained
or SeamlessM4T fails to load. Uses the facebook/nllb-200-distilled-1.3B model.

NLLB language codes differ from SeamlessM4T:
  Nepali → npi_Deva (Devanagari script)
  English → eng_Latn
  Hindi   → hin_Deva
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import structlog

from server.core.config import TranslationConfig
from server.translation.engine import TranslationEngine, TranslationResult

logger = structlog.get_logger(__name__)

# SeamlessM4T ISO 639-3 → NLLB BCP-47 with script tag
_LANG_MAP: dict[str, str] = {
    "npi": "npi_Deva",
    "eng": "eng_Latn",
    "hin": "hin_Deva",
}


class NLLBTranslator(TranslationEngine):
    """NLLB-200 distilled-1.3B via Hugging Face transformers."""

    MODEL_ID = "facebook/nllb-200-distilled-1.3B"

    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self._tokenizer = None
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nllb")
        self._loaded = False
        self._cache: dict[tuple[str, str], str] = {}
        self._cache_order: list[tuple[str, str]] = []

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        t0 = time.perf_counter()
        dtype = torch.float16 if self._config.torch_dtype == "float16" else torch.float32

        model_id = self._config.model_path or self.MODEL_ID
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if self._config.device == "cuda":
            self._model = self._model.cuda()

        self._model.eval()
        self._loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("NLLB-200 loaded", model=model_id, load_ms=round(elapsed))

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
    ) -> TranslationResult:
        if not self._loaded:
            raise RuntimeError("NLLBTranslator not loaded — call load() first")

        text = text.strip()
        if not text:
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=0.0,
            )

        nllb_src = _LANG_MAP.get(source_lang, source_lang)
        nllb_tgt = _LANG_MAP.get(target_lang, target_lang)
        cache_key = (text.lower(), nllb_tgt)

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
            nllb_src,
            nllb_tgt,
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

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch
        assert self._tokenizer is not None and self._model is not None

        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if self._config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=self._config.num_beams,
                max_new_tokens=self._config.max_new_tokens,
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _cache_put(self, key: tuple[str, str], value: str) -> None:
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self._config.lru_cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._cache_order.append(key)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_gb(self) -> float:
        return 2.6 if self._loaded else 0.0

    @property
    def model_name(self) -> str:
        return "nllb-200-distilled-1.3b"
