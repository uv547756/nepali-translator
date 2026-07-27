"""
Incremental translation state machine.

Two cooperating classes:

  IncrementalASRState — tracks partial ASR transcript history and identifies
    words that have "stabilized" (appeared identically in the last N consecutive
    partial transcripts). Stable words are safe to send to translation without
    waiting for end-of-speech.

  IncrementalTranslationState — receives stable word groups, translates each
    group (using prior translated context for coherence), accumulates translated
    text, splits at sentence boundaries, and pushes complete sentences to the
    TTS audio queue. Tracks a 'spoken_end_char' cursor so that no text is ever
    synthesized twice.

Together these implement the incremental translation loop:
  new stable words → re-translate the whole committed prefix
  → split at boundary → speak only complete, not-yet-spoken sentences

The prefix is re-translated rather than the new fragment alone: an MT model
given an isolated fragment has no context and will confabulate one. The
spoken-text cursor is what makes repeated re-translation safe, by ensuring
nothing is ever spoken twice.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from server.translation.engine import TranslationEngine
from server.tts.engine import TTSEngine

logger = structlog.get_logger(__name__)

# ── Sentence boundary splitting ─────────────────────────────────────────────

_HARD_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\(])')
_SOFT_BOUNDARY = re.compile(r'(?<=[,;:])\s+')
_SOFT_SPLIT_THRESHOLD = 80


def _split_at_boundary(
    text: str,
    soft_threshold: int = _SOFT_SPLIT_THRESHOLD,
) -> tuple[list[str], str]:
    """Split text at sentence boundaries.

    Returns (complete_sentences, incomplete_remainder).
    Hard boundaries: [.!?] followed by a capital letter or quote.
    Soft boundaries: [,;:] — only when buffer exceeds soft_threshold.
    """
    parts = _HARD_BOUNDARY.split(text)
    if len(parts) > 1:
        return parts[:-1], parts[-1]

    if len(text) >= soft_threshold:
        parts = _SOFT_BOUNDARY.split(text, maxsplit=1)
        if len(parts) > 1:
            return [parts[0]], parts[1]

    return [], text


# ── IncrementalASRState ─────────────────────────────────────────────────────

class IncrementalASRState:
    """Identifies stable (committed) words from a stream of partial ASR results.

    A word at position i is "stable" when the last N consecutive partial
    transcripts all agree on the same string at that position.
    Stable words are guaranteed (with high probability) not to change in
    future partials from the same utterance.

    Call update() for each partial; call finalize() on end-of-speech.
    """

    def __init__(self, stability_window: int = 3) -> None:
        self._window = stability_window
        self._history: deque[list[str]] = deque(maxlen=stability_window)
        self.committed_words: list[str] = []

    def update(self, partial_words: list[str]) -> list[str]:
        """Feed the word list from a new partial transcript.

        Returns the list of newly committed words (may be empty).
        Words already in self.committed_words are never returned again.
        """
        self._history.append(partial_words)

        if len(self._history) < self._window:
            return []  # Not enough history to call anything stable yet

        # Find the longest common prefix across all history entries
        min_len = min(len(h) for h in self._history)
        stable_up_to = 0
        for i in range(min_len):
            reference = self._history[0][i]
            if all(h[i] == reference for h in self._history):
                stable_up_to = i + 1
            else:
                break

        # Newly stable = positions beyond already-committed words
        prev_committed = len(self.committed_words)
        if stable_up_to <= prev_committed:
            return []

        newly_stable = partial_words[prev_committed:stable_up_to]
        self.committed_words.extend(newly_stable)
        return newly_stable

    def finalize(self, final_words: list[str]) -> list[str]:
        """Called with the definitive final transcript at end-of-speech.

        Returns the uncommitted suffix (words not yet seen by update()).
        Resets internal state for the next utterance.
        """
        uncommitted = final_words[len(self.committed_words):]
        self.reset()
        return uncommitted

    def reset(self) -> None:
        self._history.clear()
        self.committed_words = []

    @property
    def committed_text(self) -> str:
        return " ".join(self.committed_words)


# ── IncrementalTranslationState ─────────────────────────────────────────────

class IncrementalTranslationState:
    """Manages the full committed-text → translation → TTS chain for one utterance.

    Design contract:
    - Call on_partial_words() each time IncrementalASRState.update() yields new words.
    - Call on_final_words() once when VAD detects end-of-speech.
    - After on_final_words() returns, call reset() to prepare for the next utterance.

    No-repeat guarantee:
    - '_spoken_text' holds exactly the text already sent to TTS. Each pass
      considers only the remainder past it, and emits only complete sentences,
      so a half sentence is never spoken and no sentence is spoken twice --
      even though re-translation may reword the tail between passes.
    """

    def __init__(
        self,
        asr_state: IncrementalASRState,
        translator: TranslationEngine,
        tts: TTSEngine,
        audio_output_queue: asyncio.Queue[bytes],
        source_lang: str = "npi",
        target_lang: str = "eng",
        context_window_chars: int = 200,
        min_commit_words: int = 1,
        max_buffer_chars: int = _SOFT_SPLIT_THRESHOLD,
        metrics: Any = None,
    ) -> None:
        self._asr_state = asr_state
        self._translator = translator
        self._tts = tts
        self._audio_q = audio_output_queue
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._context_window = context_window_chars
        self._min_commit_words = min_commit_words
        self._max_buffer_chars = max_buffer_chars
        # Optional MetricsCollector; TTS/translation happen here rather than in
        # the pipeline, so without this their latencies are never recorded.
        self._metrics = metrics

        # Accumulated translations for this utterance
        self._committed_translation: str = ""
        # Text already sent to TTS. Tracked as text, not an index, because
        # re-translation can reword the tail between passes.
        self._spoken_text: str = ""
        self._spoken_end_char: int = 0

        # Metrics
        self._translation_latencies: list[float] = []
        self._tts_latencies: list[float] = []

    async def on_partial_words(self, partial_words: list[str]) -> None:
        """Process a new partial transcript from ASR.

        Re-translates the whole committed prefix and speaks only the complete
        sentences that have not been spoken yet.

        Why re-translate rather than translate just the newly stable words:
        an MT model given a bare fragment has no context and will invent one.
        Feeding SeamlessM4T a lone noun produced confident nonsense of the form
        "What is the meaning of the name X?". Translating the full prefix costs
        one extra forward pass per interval and keeps the model in-context;
        the spoken_end_char cursor is what makes that safe to do repeatedly.
        """
        newly_stable = self._asr_state.update(partial_words)

        if len(newly_stable) < self._min_commit_words:
            return

        full_source = " ".join(self._asr_state.committed_words).strip()
        if not full_source:
            return

        t0 = time.perf_counter()
        try:
            result = await self._translator.translate(
                full_source,
                self._source_lang,
                self._target_lang,
                context="",
            )
            self._translation_latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.error("Translation failed in on_partial_words", error=str(exc))
            return

        translation = (result.translated_text or "").strip()
        if not translation:
            return

        self._committed_translation = translation
        await self._speak_new_sentences(translation)

    async def _speak_new_sentences(self, translation: str) -> None:
        """Speak complete sentences in `translation` beyond what was already spoken.

        Re-translation can reword the tail between passes, so the already-spoken
        prefix is tracked as text rather than an index: only the remainder past
        it is considered, and only whole sentences are emitted. A half sentence
        is never spoken, since it may be reworded on the next pass.
        """
        if translation.startswith(self._spoken_text):
            remainder = translation[len(self._spoken_text):]
        elif len(translation) > len(self._spoken_text):
            # Prefix was reworded. Do not re-speak; take only the extra tail.
            remainder = translation[len(self._spoken_text):]
        else:
            return

        remainder = remainder.lstrip()
        if not remainder:
            return

        complete, _incomplete = _split_at_boundary(remainder, self._max_buffer_chars)
        for sentence in complete:
            sentence = sentence.strip()
            if not sentence:
                continue
            await self._synthesize_and_queue(sentence)
            self._spoken_text = (
                f"{self._spoken_text} {sentence}".strip()
                if self._spoken_text else sentence
            )
            self._spoken_end_char = len(self._spoken_text)

    async def _synthesize_and_queue(self, sentence: str) -> None:
        """Synthesize one sentence and queue its PCM."""
        t0 = time.perf_counter()
        try:
            pcm = await self._tts_synthesize(sentence)
            tts_ms = (time.perf_counter() - t0) * 1000
            self._tts_latencies.append(tts_ms)
            if self._metrics is not None:
                self._metrics.record_tts(tts_ms)
            await self._audio_q.put(pcm)
        except Exception as exc:
            logger.error("TTS synthesis failed", sentence=sentence[:50], error=str(exc))

    async def on_final_words(self, final_words: list[str]) -> None:
        """Process the final definitive transcript at end-of-speech.

        Translates any uncommitted suffix and flushes the entire sentence buffer.
        """
        # finalize() only returns the uncommitted suffix, but the whole
        # utterance is translated here as one unit: this is the definitive
        # transcript, and a full-sentence input is what the MT model translates
        # best. The spoken-text cursor prevents anything being said twice.
        self._asr_state.finalize(final_words)

        full_source = " ".join(final_words).strip()
        if full_source:
            t0 = time.perf_counter()
            try:
                result = await self._translator.translate(
                    full_source,
                    self._source_lang,
                    self._target_lang,
                    context="",
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                self._translation_latencies.append(latency_ms)

                logger.info(
                    "Translation result",
                    source=full_source,
                    translated=result.translated_text,
                    latency_ms=round(latency_ms),
                )

                translation = (result.translated_text or "").strip()
                if translation:
                    self._committed_translation = translation
                    # Whole sentences first...
                    await self._speak_new_sentences(translation)
                    # ...then whatever tail remains, since end-of-speech means
                    # no further words are coming to complete it.
                    tail = translation[len(self._spoken_text):].strip()
                    if tail:
                        await self._synthesize_and_queue(tail)
                        self._spoken_text = translation
                        self._spoken_end_char = len(translation)

            except Exception as exc:
                logger.error(
                    "Translation failed in on_final_words",
                    error=str(exc),
                    exc_info=True,
                )

        # Signal end of utterance
        await self._audio_q.put(b"")

        logger.debug(
            "Utterance complete",
            source_words=len(final_words),
            translation_latency_avg_ms=round(
                sum(self._translation_latencies) / max(len(self._translation_latencies), 1), 1
            ),
        )

    async def _tts_synthesize(self, text: str) -> bytes:
        """Synthesize text and return Int16 LE PCM bytes."""
        import numpy as np
        result = await self._tts.synthesize(text)
        int16 = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)
        return int16.tobytes()

    def reset(self) -> None:
        """Full reset for the next utterance."""
        self._asr_state.reset()
        self._committed_translation = ""
        self._spoken_text = ""
        self._spoken_end_char = 0
        self._translation_latencies = []
        self._tts_latencies = []

    @property
    def committed_translation(self) -> str:
        return self._committed_translation

    @property
    def avg_translation_latency_ms(self) -> float:
        if not self._translation_latencies:
            return 0.0
        return sum(self._translation_latencies) / len(self._translation_latencies)

    @property
    def avg_tts_latency_ms(self) -> float:
        if not self._tts_latencies:
            return 0.0
        return sum(self._tts_latencies) / len(self._tts_latencies)
