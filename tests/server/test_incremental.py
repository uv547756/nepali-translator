"""
Unit tests for the incremental ASR state machine.

Tests the word stability tracking algorithm without requiring
any model to be loaded.
"""

import pytest

from server.core.incremental import IncrementalASRState, _split_at_boundary


# ── IncrementalASRState ─────────────────────────────────────────────────────

def test_no_stability_with_insufficient_history():
    state = IncrementalASRState(stability_window=3)
    assert state.update(["तपाईं", "कहाँ"]) == []
    assert state.update(["तपाईं", "कहाँ"]) == []
    # Only 2 history entries — need 3


def test_words_committed_after_window():
    state = IncrementalASRState(stability_window=3)
    state.update(["तपाईं", "कहाँ"])
    state.update(["तपाईं", "कहाँ"])
    new = state.update(["तपाईं", "कहाँ"])
    assert new == ["तपाईं", "कहाँ"]
    assert state.committed_words == ["तपाईं", "कहाँ"]


def test_unstable_word_not_committed():
    state = IncrementalASRState(stability_window=3)
    state.update(["म", "जान्छु", "A"])
    state.update(["म", "जान्छु", "B"])
    new = state.update(["म", "जान्छु", "C"])
    # "म" and "जान्छु" are stable but "A/B/C" differ
    assert new == ["म", "जान्छु"]


def test_incremental_growth():
    state = IncrementalASRState(stability_window=3)
    state.update(["म", "जान्छु"])
    state.update(["म", "जान्छु", "बजार"])
    new1 = state.update(["म", "जान्छु", "बजार"])
    assert new1 == ["म", "जान्छु"]   # "बजार" only appeared twice in last 3

    state.update(["म", "जान्छु", "बजार", "आज"])
    new2 = state.update(["म", "जान्छु", "बजार", "आज"])
    assert "बजार" in new2


def test_finalize_returns_uncommitted():
    state = IncrementalASRState(stability_window=3)
    state.update(["म", "जान्छु"])
    state.update(["म", "जान्छु"])
    state.update(["म", "जान्छु"])
    # "म" and "जान्छु" are committed

    uncommitted = state.finalize(["म", "जान्छु", "किन्न"])
    assert uncommitted == ["किन्न"]


def test_finalize_resets_state():
    state = IncrementalASRState(stability_window=3)
    for _ in range(3):
        state.update(["hello"])
    assert state.committed_words == ["hello"]
    state.finalize(["hello"])
    assert state.committed_words == []
    assert list(state._history) == []


def test_words_not_recommitted():
    state = IncrementalASRState(stability_window=2)
    state.update(["A", "B"])
    new1 = state.update(["A", "B", "C"])
    assert "A" in new1 and "B" in new1

    new2 = state.update(["A", "B", "C"])
    # A and B already committed — should not appear again
    assert "A" not in new2 and "B" not in new2


# ── Sentence boundary splitter ──────────────────────────────────────────────

def test_hard_boundary_splits():
    sentences, remainder = _split_at_boundary("Hello world. How are you?")
    assert sentences == ["Hello world."]
    assert remainder == "How are you?"


def test_no_split_short_text():
    sentences, remainder = _split_at_boundary("Hello, how are you")
    assert sentences == []
    assert remainder == "Hello, how are you"


def test_soft_boundary_long_text():
    # Must exceed _SOFT_SPLIT_THRESHOLD (80) for a soft split to trigger.
    # The original fixture was 79 chars, so this assertion never actually held.
    text = (
        "I went to the market, but it was closed, "
        "so I went home instead and made dinner for everyone"
    )
    assert len(text) >= 80, "fixture must exceed the soft-split threshold"
    sentences, remainder = _split_at_boundary(text)
    assert len(sentences) == 1   # first comma triggers soft split
    assert len(remainder) > 0


def test_soft_boundary_not_triggered_below_threshold():
    text = "a, " + "b" * 60          # comma present, but under the threshold
    assert len(text) < 80
    sentences, remainder = _split_at_boundary(text)
    assert sentences == []
    assert remainder == text


def test_soft_threshold_is_configurable():
    text = "aaa, bbb"
    assert _split_at_boundary(text) == ([], text)          # default 80: no split
    assert _split_at_boundary(text, 5) == (["aaa,"], "bbb")  # lowered: splits


def test_multiple_hard_boundaries():
    text = "First sentence. Second sentence. Third part"
    sentences, remainder = _split_at_boundary(text)
    assert len(sentences) == 2
    assert remainder == "Third part"


def test_question_mark_boundary():
    text = "Where are you going? I am going home."
    sentences, remainder = _split_at_boundary(text)
    assert sentences[0] == "Where are you going?"
