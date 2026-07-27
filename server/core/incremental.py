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
  new stable words → translate chunk → append to sentence buffer
  → split at boundary → push complete sentences to TTS → advance cursor
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator

import structlog

from server.translation.engine import TranslationEngine
from server.tts.engine import TTSEngine

logger = structlog.get_logger(__name__)

# ── Sentence boundary splitting ─────────────────────────────────────────────

_HARD_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\(])')
_SOFT_BOUNDARY = re.compile(r'(?<=[,;:])\s+')
_SOFT_SPLIT_THRESHOLD = 80


def _split_at_boundary(text: str) -> tuple[list[str], str]:
    """Split text at sentence boundaries.

    Returns (complete_sentences, incomplete_remainder).
    Hard boundaries: [.!?] followed by a capital letter or quote.
    Soft boundaries: [,;:] — only when buffer exceeds threshold.
    """
    parts = _HARD_BOUNDARY.split(text)
    if len(parts) > 1:
        return parts[:-1], parts[-1]

    if len(text) >= _SOFT_SPLIT_THRESHOLD:
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
    - 'spoken_end_char' tracks how many bytes of committed_translation have been
      pushed to TTS. Every _flush_to_tts() call only processes NEW text beyond
      this cursor. Prior translations are never re-processed.
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
    ) -> None:
        self._asr_state = asr_state
        self._translator = translator
        self._tts = tts
        self._audio_q = audio_output_queue
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._context_window = context_window_chars
        self._min_commit_words = min_commit_words

        # Accumulated translations for this utterance
        self._committed_translation: str = ""
        self._sentence_buffer: str = ""
        self._spoken_end_char: int = 0

        # Metrics
        self._translation_latencies: list[float] = []
        self._tts_latencies: list[float] = []

    async def on_partial_words(self, partial_words: list[str]) -> None:
        """Process a new partial transcript from ASR.

        Feeds words to IncrementalASRState, translates newly stable words,
        and flushes complete sentences to TTS.
        """
        newly_stable = self._asr_state.update(partial_words)

        if len(newly_stable) < self._min_commit_words:
            return

        source_chunk = " ".join(newly_stable)
        context = self._committed_translation[-self._context_window:]

        t0 = time.perf_counter()
        try:
            result = await self._translator.translate(
                source_chunk,
                self._source_lang,
                self._target_lang,
                context=context,
            )
            self._translation_latencies.append((time.perf_counter() - t0) * 1000)

            if result.translated_text:
                await self._flush_to_tts(result.translated_text)

        except Exception as exc:
            logger.error("Translation failed in on_partial_words", error=str(exc))

    async def on_final_words(self, final_words: list[str]) -> None:
        """Process the final definitive transcript at end-of-speech.

        Translates any uncommitted suffix and flushes the entire sentence buffer.
        """
        uncommitted = self._asr_state.finalize(final_words)

        if uncommitted:
            source_chunk = " ".join(uncommitted)
            context = self._committed_translation[-self._context_window:]

            t0 = time.perf_counter()
            try:
                result = await self._translator.translate(
                    source_chunk,
                    self._source_lang,
                    self._target_lang,
                    context=context,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                self._translation_latencies.append(latency_ms)

                logger.info(
                    "Translation result",
                    source=source_chunk,
                    translated=result.translated_text,
                    latency_ms=round(latency_ms),
                )

                if result.translated_text:
                    self._committed_translation += (
                        " " + result.translated_text if self._committed_translation
                        else result.translated_text
                    )

            except Exception as exc:
                logger.error(
                    "Translation failed in on_final_words",
                    error=str(exc),
                    exc_info=True,
                )

        # Flush remaining sentence buffer, ignoring boundary — it's end-of-utterance
        remainder = self._sentence_buffer.strip()
        if remainder:
            t0 = time.perf_counter()
            try:
                pcm = await self._tts_synthesize(remainder)
                self._tts_latencies.append((time.perf_counter() - t0) * 1000)
                await self._audio_q.put(pcm)
                self._spoken_end_char += len(remainder)
            except Exception as exc:
                logger.error("TTS failed for sentence buffer", error=str(exc))

        # Signal end of utterance
        await self._audio_q.put(b"")

        logger.debug(
            "Utterance complete",
            source_words=len(final_words),
            translation_latency_avg_ms=round(
                sum(self._translation_latencies) / max(len(self._translation_latencies), 1), 1
            ),
        )

    async def _flush_to_tts(self, new_translation: str) -> None:
        """Append new_translation to the sentence buffer and push complete sentences."""
        self._committed_translation += (
            " " + new_translation if self._committed_translation else new_translation
        )
        self._sentence_buffer += (
            " " + new_translation if self._sentence_buffer else new_translation
        )

        complete_sentences, self._sentence_buffer = _split_at_boundary(self._sentence_buffer)

        for sentence in complete_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            t0 = time.perf_counter()
            try:
                pcm = await self._tts_synthesize(sentence)
                self._tts_latencies.append((time.perf_counter() - t0) * 1000)
                await self._audio_q.put(pcm)
                self._spoken_end_char += len(sentence)
            except Exception as exc:
                logger.error("TTS synthesis failed", sentence=sentence[:50], error=str(exc))

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
        self._sentence_buffer = ""
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
