"""Unit tests for the speaker-change boundary.

The embedder is a stub that reads the amplitude of the audio, so a "voice"
here is a volume. That is enough to test the policy - when a comparison is
made at all, what it is compared against, and where the cut lands - which is
the part that has to be right before a model is involved.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import SAMPLE_RATE, SAMPLE_WIDTH
from server.pipeline.buffer import PartialWindow
from server.pipeline.speaker_change import SpeakerChangeDetector

WINDOW_MS = 1_000


class VoiceByVolume:
    """A voiceprint that is just the mean amplitude, as a unit vector."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        level = float(np.abs(samples).mean())
        # Two voices: loud points one way, quiet the other.
        return np.array([1.0, 0.0]) if level > 4_000 else np.array([0.0, 1.0])


def tone(ms: float, amplitude: int) -> bytes:
    samples = np.full(int(ms * SAMPLE_RATE / 1000.0), amplitude, dtype="<i2")
    return samples.tobytes()


LOUD = 8_000
QUIET = 500


def window(pcm: bytes, index: int = 0) -> PartialWindow:
    return PartialWindow(index=index, pcm=pcm, start_ms=0.0)


def detector(**kwargs) -> SpeakerChangeDetector:
    defaults = dict(embedder=VoiceByVolume(), window_ms=WINDOW_MS,
                    threshold=0.25)
    defaults.update(kwargs)
    return SpeakerChangeDetector(**defaults)


# ---------------------------------------------------------------------------
# When it looks at all
# ---------------------------------------------------------------------------
def test_a_short_utterance_is_left_alone():
    """Under two windows there is nowhere to put a comparison."""
    watcher = detector()
    assert watcher.observe(window(tone(1_500, LOUD))) is None
    assert watcher.embedder.calls == 0, "it embedded audio it could not use"


def test_one_voice_throughout_is_not_a_change():
    assert detector().observe(window(tone(2_400, LOUD))) is None


def test_a_second_voice_taking_over_is_a_change():
    change = detector().observe(window(tone(1_400, LOUD) + tone(1_000, QUIET)))
    assert change is not None


# ---------------------------------------------------------------------------
# Where the cut goes
# ---------------------------------------------------------------------------
def test_the_cut_is_placed_before_the_window_that_disagreed():
    """Not where it was noticed: that would hand the newcomer's second to
    whoever was speaking before."""
    change = detector().observe(window(tone(1_400, LOUD) + tone(1_000, QUIET)))
    assert change.at_ms == pytest.approx(1_400, abs=1)


def test_the_similarity_that_decided_it_is_reported():
    change = detector().observe(window(tone(1_400, LOUD) + tone(1_000, QUIET)))
    assert change.similarity < 0.25


# ---------------------------------------------------------------------------
# What it compares against
# ---------------------------------------------------------------------------
def test_the_anchor_is_whoever_opened_the_utterance():
    """Not the previous window - a slow speaker would drift out of themselves
    one window at a time."""
    watcher = detector()
    watcher.observe(window(tone(2_400, LOUD)))
    before = watcher.embedder.calls
    watcher.observe(window(tone(3_000, LOUD)))
    assert watcher.embedder.calls == before + 1, "the anchor was re-embedded"


def test_a_new_utterance_gets_a_new_anchor():
    watcher = detector()
    watcher.observe(window(tone(2_400, LOUD), index=0))
    # The next utterance is the quiet voice throughout; against the old anchor
    # every window of it would look like a change.
    assert watcher.observe(window(tone(2_400, QUIET), index=1)) is None


def test_after_a_change_the_new_voice_becomes_the_anchor():
    """Otherwise the same change is reported again on every later partial."""
    watcher = detector()
    pcm = tone(1_400, LOUD) + tone(1_000, QUIET)
    assert watcher.observe(window(pcm)) is not None
    assert watcher.observe(window(pcm + tone(600, QUIET))) is None


def test_reset_forgets_the_meeting():
    watcher = detector()
    watcher.observe(window(tone(2_400, LOUD), index=0))
    watcher.reset()
    assert watcher.stats.checks == 0
    assert watcher.observe(window(tone(2_400, QUIET), index=0)) is None


# ---------------------------------------------------------------------------
# Evidence for tuning
# ---------------------------------------------------------------------------
def test_every_comparison_is_recorded():
    """The threshold was set from eleven pairs; a real meeting has to be able
    to say where the two ranges actually sit."""
    watcher = detector()
    watcher.observe(window(tone(2_400, LOUD)))
    watcher.observe(window(tone(1_400, LOUD) + tone(1_000, QUIET)))
    assert watcher.stats.checks == 2
    assert watcher.stats.changes == 1
    assert len(watcher.stats.scores) == 2


def test_a_threshold_outside_cosine_range_is_refused():
    with pytest.raises(ValueError):
        detector(threshold=1.5)


# ---------------------------------------------------------------------------
# A model that answers badly
# ---------------------------------------------------------------------------
class NoVoice:
    def embed(self, pcm: bytes) -> np.ndarray:
        return np.array([])


class NotFinite:
    def embed(self, pcm: bytes) -> np.ndarray:
        return np.array([np.nan, 1.0])


@pytest.mark.parametrize("embedder", [NoVoice(), NotFinite()])
def test_an_unusable_voiceprint_cuts_nothing(embedder):
    """A bad embedding must not become a sentence boundary."""
    watcher = detector(embedder=embedder)
    assert watcher.observe(window(tone(2_400, LOUD))) is None


def test_the_window_is_measured_in_whole_samples():
    assert detector().window_bytes == WINDOW_MS * SAMPLE_RATE // 1000 * SAMPLE_WIDTH
