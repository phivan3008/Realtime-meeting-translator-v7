"""Unit tests for the Deep Noise Filter policy.

YAMNet itself is replaced by a stub, so these run on the Dev PC without
TensorFlow. The model's real behaviour is covered by
``server/tests_real/test_real_noise.py`` on the GPU pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.pipeline.noise import (
    NOISE_LABELS,
    SPEECH_LABELS,
    Classification,
    NoiseFilter,
    aggregate,
    top_labels,
)


class StubClassifier:
    """Returns a scripted Classification and remembers what it was given."""

    def __init__(self, *results: Classification):
        self.results = list(results)
        self.calls: list[bytes] = []

    def classify(self, pcm: bytes) -> Classification:
        self.calls.append(pcm)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


def speechy(score: float = 0.8, noise: float = 0.1) -> Classification:
    return Classification(speech_score=score, noise_score=noise,
                          noise_label="Typing",
                          top=(("Speech", score), ("Typing", noise)))


def noisy(noise: float = 0.7, label: str = "Computer keyboard",
          speech: float = 0.01) -> Classification:
    return Classification(speech_score=speech, noise_score=noise,
                          noise_label=label,
                          top=((label, noise), ("Speech", speech)))


def audio(seconds: float = 1.0) -> bytes:
    return bytes(int(seconds * 16_000) * 2)


def make_filter(*results: Classification, **kwargs) -> NoiseFilter:
    return NoiseFilter(classifier=StubClassifier(*results), **kwargs)


# ---------------------------------------------------------------------------
# The label sets
# ---------------------------------------------------------------------------
def test_the_two_label_sets_do_not_overlap():
    assert SPEECH_LABELS & NOISE_LABELS == frozenset()


def test_the_noise_set_covers_what_design_md_names():
    """Keyboard clatter and coughing are the two examples in the design."""
    assert "Computer keyboard" in NOISE_LABELS
    assert "Typing" in NOISE_LABELS
    assert "Cough" in NOISE_LABELS


def test_ambient_room_labels_are_not_treated_as_noise_evidence():
    """They score high under speech too, so they prove nothing."""
    for label in ("Inside, small room", "Inside, large room or hall"):
        assert label not in NOISE_LABELS


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------
def test_aggregate_takes_the_best_frame_not_the_average():
    """One clear sentence inside seven seconds of typing still counts."""
    labels = ["Speech", "Typing"]
    scores = np.array([[0.01, 0.9], [0.01, 0.9], [0.85, 0.2]])
    assert aggregate(scores, labels, {"Speech"}) == (pytest.approx(0.85), "Speech")


def test_aggregate_names_the_label_that_won():
    labels = ["Typing", "Cough"]
    scores = np.array([[0.3, 0.7]])
    score, label = aggregate(scores, labels, {"Typing", "Cough"})
    assert (round(score, 2), label) == (0.7, "Cough")


def test_aggregate_ignores_labels_outside_the_set():
    labels = ["Speech", "Music"]
    scores = np.array([[0.1, 0.99]])
    assert aggregate(scores, labels, {"Speech"})[0] == pytest.approx(0.1)


def test_aggregate_returns_zero_when_nothing_matches():
    assert aggregate(np.array([[0.5]]), ["Music"], {"Speech"}) == (0.0, "")


def test_top_labels_are_ordered_by_peak_score():
    labels = ["A", "B", "C"]
    scores = np.array([[0.1, 0.9, 0.5], [0.2, 0.3, 0.4]])
    assert top_labels(scores, labels, count=2) == (("B", pytest.approx(0.9)),
                                                   ("C", pytest.approx(0.5)))


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_speech_is_kept():
    verdict = make_filter(speechy()).judge(audio())
    assert verdict.keep is True
    assert verdict.reason == "speech detected"


def test_keyboard_clatter_is_dropped():
    verdict = make_filter(noisy(label="Computer keyboard")).judge(audio())
    assert verdict.keep is False
    assert "Computer keyboard" in verdict.reason


def test_a_cough_is_dropped():
    verdict = make_filter(noisy(label="Cough")).judge(audio())
    assert verdict.keep is False
    assert "Cough" in verdict.reason


def test_a_borderline_utterance_is_kept():
    """Right on the threshold counts as speech, because losing it is worse."""
    verdict = make_filter(speechy(score=0.2, noise=0.9)).judge(audio())
    assert verdict.keep is True


def test_just_under_the_threshold_with_loud_noise_is_dropped():
    verdict = make_filter(noisy(speech=0.19, noise=0.9)).judge(audio())
    assert verdict.keep is False


def test_unrecognisable_audio_is_kept_rather_than_guessed_away():
    """No speech, no noise either: Whisper decides, not us."""
    verdict = make_filter(Classification(0.02, 0.01)).judge(audio())
    assert verdict.keep is True
    assert "nothing conclusive" in verdict.reason


def test_the_timid_rule_can_be_turned_off():
    verdict = make_filter(
        Classification(0.02, 0.01, noise_label="Silence"),
        require_louder_noise=False,
    ).judge(audio())
    assert verdict.keep is False


def test_empty_audio_is_dropped_without_calling_the_model():
    stub = StubClassifier(speechy())
    verdict = NoiseFilter(classifier=stub).judge(b"")
    assert verdict.keep is False
    assert verdict.reason == "empty audio"
    assert stub.calls == []


def test_the_whole_utterance_is_handed_to_the_classifier():
    stub = StubClassifier(speechy())
    pcm = audio(2.0)
    NoiseFilter(classifier=stub).judge(pcm)
    assert stub.calls == [pcm]


def test_the_threshold_is_configurable():
    strict = make_filter(noisy(speech=0.3, noise=0.9), min_speech_score=0.5)
    assert strict.judge(audio()).keep is False
    lenient = make_filter(noisy(speech=0.3, noise=0.9), min_speech_score=0.1)
    assert lenient.judge(audio()).keep is True


def test_an_out_of_range_threshold_is_rejected():
    with pytest.raises(ValueError, match="min_speech_score"):
        make_filter(speechy(), min_speech_score=1.5)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def test_stats_count_what_was_kept_and_dropped():
    filt = make_filter(speechy(), noisy(), noisy(label="Cough"))
    for _ in range(3):
        filt.judge(audio())
    assert filt.stats.seen == 3
    assert filt.stats.kept == 1
    assert filt.stats.dropped == 2
    assert filt.stats.dropped_labels == {"Computer keyboard": 1, "Cough": 1}


def test_reset_clears_the_counters():
    filt = make_filter(noisy())
    filt.judge(audio())
    filt.reset()
    assert filt.stats.seen == 0
    assert filt.stats.dropped_labels == {}


def test_a_classification_summarises_itself_for_the_log():
    summary = speechy(0.77).summary()
    assert "speech 0.77" in summary
    assert "Speech" in summary


def test_top_label_is_empty_when_nothing_was_scored():
    assert Classification(0.0, 0.0).top_label == ""
