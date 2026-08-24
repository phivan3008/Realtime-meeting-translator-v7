"""Unit tests for the two-way language decision.

The model is stubbed, so these run without torch or a checkpoint. Whether
VoxLingua107 actually tells Vietnamese from Japanese on real audio is a
question for ``server/tests_real/test_real_lid.py`` on the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (
    LID_LANGUAGES,
    LID_UNKNOWN,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.lid import (
    LanguageIdentifier,
    language_code,
    two_way_probabilities,
)


class StubScorer:
    """Returns scripted log-probabilities per language."""

    def __init__(self, *rounds: dict):
        self.rounds = list(rounds)
        self.calls = 0

    def scores(self, pcm: bytes) -> dict:
        value = self.rounds[min(self.calls, len(self.rounds) - 1)]
        self.calls += 1
        return dict(value)


def audio(ms: float) -> bytes:
    return bytes(int(ms * SAMPLE_RATE / 1000) * SAMPLE_WIDTH)


def make_identifier(*rounds: dict, **kwargs) -> LanguageIdentifier:
    defaults = dict(scorer=StubScorer(*rounds), min_margin=0.30,
                    min_duration_ms=600)
    defaults.update(kwargs)
    return LanguageIdentifier(**defaults)


# Log-probabilities, as the model produces.
CLEARLY_VI = {"vi": math.log(0.90), "ja": math.log(0.02)}
CLEARLY_JA = {"vi": math.log(0.01), "ja": math.log(0.85)}
AMBIGUOUS = {"vi": math.log(0.30), "ja": math.log(0.28)}


# ---------------------------------------------------------------------------
# Labels and probabilities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label, expected", [
    ("vi", "vi"),
    ("vi: Vietnamese", "vi"),
    (" JA : Japanese ", "ja"),
    ("ja", "ja"),
])
def test_a_label_reduces_to_its_language_code(label, expected):
    """The checkpoint spells these differently between revisions."""
    assert language_code(label) == expected


def test_probabilities_are_renormalised_over_the_two_languages():
    """The question is which of these two, not which of 107."""
    probabilities = two_way_probabilities({"vi": math.log(0.10),
                                           "ja": math.log(0.30)})
    assert probabilities["ja"] == pytest.approx(0.75)
    assert probabilities["vi"] == pytest.approx(0.25)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_equal_scores_split_evenly():
    probabilities = two_way_probabilities({"vi": -1.0, "ja": -1.0})
    assert probabilities == {"vi": pytest.approx(0.5), "ja": pytest.approx(0.5)}


def test_very_small_scores_do_not_overflow():
    """exp() of a large negative log-prob is where naive code dies."""
    probabilities = two_way_probabilities({"vi": -800.0, "ja": -805.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["vi"] > probabilities["ja"]


def test_no_scores_gives_no_probabilities():
    assert two_way_probabilities({}) == {}


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def test_the_meeting_languages_are_the_two_from_design_md():
    assert set(LID_LANGUAGES) == {"vi", "ja"}


def test_clear_vietnamese_is_reported_as_vietnamese():
    decision = make_identifier(CLEARLY_VI).identify(audio(2000))
    assert decision.lang_code == "vi"
    assert decision.known is True
    assert decision.confidence > 0.9
    assert decision.margin > 0.8


def test_clear_japanese_is_reported_as_japanese():
    assert make_identifier(CLEARLY_JA).identify(audio(2000)).lang_code == "ja"


def test_two_close_languages_are_reported_as_unknown():
    """Forced Japanese on Vietnamese audio transcribes confident nonsense."""
    decision = make_identifier(AMBIGUOUS).identify(audio(2000))
    assert decision.lang_code == LID_UNKNOWN
    assert decision.known is False
    assert "letting the ASR detect it" in decision.reason


def test_the_margin_is_configurable():
    close = {"vi": math.log(0.60), "ja": math.log(0.40)}
    assert make_identifier(close, min_margin=0.10).identify(audio(2000)).lang_code == "vi"
    assert make_identifier(close, min_margin=0.50).identify(audio(2000)).known is False


def test_a_short_utterance_is_not_guessed_at():
    stub = StubScorer(CLEARLY_VI)
    decision = make_identifier(scorer=stub).identify(audio(300))
    assert decision.known is False
    assert "too short" in decision.reason
    assert stub.calls == 0, "the model should not even be asked"


def test_the_short_cutoff_is_configurable():
    identifier = make_identifier(CLEARLY_VI, min_duration_ms=100)
    assert identifier.identify(audio(300)).lang_code == "vi"


def test_a_language_the_model_did_not_report_is_not_invented():
    decision = make_identifier({"vi": math.log(0.9)}).identify(audio(2000))
    assert decision.known is False
    assert "reported nothing for ['ja']" in decision.reason


def test_the_decision_carries_the_probabilities_it_used():
    decision = make_identifier(CLEARLY_VI).identify(audio(2000))
    assert set(decision.probabilities) == {"vi", "ja"}
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)


def test_only_the_wanted_languages_are_considered():
    """A Japanese sentence scored as Korean must not come back as Korean."""
    decision = make_identifier(
        {"vi": math.log(0.02), "ja": math.log(0.30), "ko": math.log(0.60)}
    ).identify(audio(2000))
    assert decision.lang_code == "ja"


@pytest.mark.parametrize("kwargs", [
    {"languages": ("vi",)},
    {"min_margin": 1.5},
    {"min_margin": -0.1},
])
def test_nonsense_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_identifier(CLEARLY_VI, **kwargs)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def test_stats_count_each_language_and_the_unknowns():
    identifier = make_identifier(CLEARLY_VI, CLEARLY_JA, AMBIGUOUS)
    for _ in range(3):
        identifier.identify(audio(2000))
    assert identifier.stats.seen == 3
    assert identifier.stats.decided == 2
    assert identifier.stats.unknown == 1
    assert identifier.stats.per_language == {"vi": 1, "ja": 1, "unknown": 1}


def test_reset_clears_the_counters():
    identifier = make_identifier(CLEARLY_VI)
    identifier.identify(audio(2000))
    identifier.reset()
    assert identifier.stats.seen == 0
