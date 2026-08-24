"""Unit tests for speaker identification.

The embedding model is stubbed, so these run without torch or a checkpoint.
Whether the real voiceprints actually separate two people is a question only
real recordings can answer - see
``server/tests_real/test_real_diarization.py``.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import SAMPLE_RATE, SAMPLE_WIDTH, SPEAKER_UNKNOWN
from server.pipeline.diarization import (
    SpeakerIdentifier,
    SpeakerRegistry,
    cosine_similarity,
    label_for,
)


class StubEmbedder:
    """Returns a scripted voiceprint per call."""

    def __init__(self, *embeddings: np.ndarray):
        self.embeddings = [np.asarray(e, dtype=np.float64) for e in embeddings]
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        value = self.embeddings[min(self.calls, len(self.embeddings) - 1)]
        self.calls += 1
        return value


def voice(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


ALICE = voice(1.0, 0.0, 0.0)
ALICE_AGAIN = voice(0.95, 0.1, 0.05)
BOB = voice(0.0, 1.0, 0.0)
CARLA = voice(0.0, 0.0, 1.0)


def audio(ms: float) -> bytes:
    return bytes(int(ms * SAMPLE_RATE / 1000) * SAMPLE_WIDTH)


# ---------------------------------------------------------------------------
# Similarity and naming
# ---------------------------------------------------------------------------
def test_a_voiceprint_matches_itself_perfectly():
    assert cosine_similarity(ALICE, ALICE) == pytest.approx(1.0)


def test_unrelated_voiceprints_score_zero():
    assert cosine_similarity(ALICE, BOB) == pytest.approx(0.0)


def test_similarity_ignores_how_loud_the_voiceprint_is():
    assert cosine_similarity(ALICE, ALICE * 7.0) == pytest.approx(1.0)


def test_an_empty_voiceprint_matches_nothing_rather_than_dividing_by_zero():
    assert cosine_similarity(ALICE, np.zeros(3)) == 0.0


def test_labels_are_the_ones_design_md_names():
    assert label_for(1) == "Speaker_01"
    assert label_for(12) == "Speaker_12"


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_the_first_voice_becomes_speaker_one():
    assignment = SpeakerRegistry().assign(ALICE)
    assert assignment.speaker_id == "Speaker_01"
    assert assignment.is_new is True


def test_the_same_voice_again_is_the_same_speaker():
    registry = SpeakerRegistry(match_threshold=0.5)
    registry.assign(ALICE)
    assignment = registry.assign(ALICE_AGAIN)
    assert assignment.speaker_id == "Speaker_01"
    assert assignment.is_new is False
    assert assignment.similarity > 0.9
    assert registry.count == 1


def test_a_different_voice_becomes_a_new_speaker():
    registry = SpeakerRegistry(match_threshold=0.5)
    registry.assign(ALICE)
    assignment = registry.assign(BOB)
    assert assignment.speaker_id == "Speaker_02"
    assert assignment.is_new is True
    assert registry.count == 2


def test_speakers_are_told_apart_across_a_conversation():
    registry = SpeakerRegistry(match_threshold=0.5)
    turns = [ALICE, BOB, ALICE_AGAIN, BOB, CARLA, ALICE]
    labels = [registry.assign(v).speaker_id for v in turns]
    assert labels == ["Speaker_01", "Speaker_02", "Speaker_01",
                      "Speaker_02", "Speaker_03", "Speaker_01"]


def test_a_higher_threshold_splits_one_voice_into_two():
    """The threshold is the whole behaviour; this is what moving it does."""
    lenient = SpeakerRegistry(match_threshold=0.5)
    lenient.assign(ALICE)
    assert lenient.assign(ALICE_AGAIN).speaker_id == "Speaker_01"

    strict = SpeakerRegistry(match_threshold=0.999)
    strict.assign(ALICE)
    assert strict.assign(ALICE_AGAIN).speaker_id == "Speaker_02"


def test_the_centroid_moves_towards_a_new_sample_but_not_all_the_way():
    registry = SpeakerRegistry(match_threshold=0.5, momentum=0.7)
    registry.assign(voice(1.0, 0.0, 0.0))
    registry.assign(voice(1.0, 1.0, 0.0))
    centroid = registry.speakers[0].centroid
    assert centroid[1] == pytest.approx(0.3)
    assert registry.speakers[0].utterances == 2


def test_one_odd_sentence_does_not_redefine_a_speaker():
    registry = SpeakerRegistry(match_threshold=0.5, momentum=0.9)
    registry.assign(ALICE)
    registry.assign(voice(0.6, 0.8, 0.0))
    assert cosine_similarity(registry.speakers[0].centroid, ALICE) > 0.95


def test_the_speaker_limit_stops_new_voices_being_invented():
    registry = SpeakerRegistry(match_threshold=0.9, max_speakers=2)
    registry.assign(ALICE)
    registry.assign(BOB)
    assignment = registry.assign(CARLA)
    assert registry.count == 2
    assert assignment.is_new is False
    assert "limit" in assignment.reason


def test_reset_forgets_everyone_for_the_next_meeting():
    registry = SpeakerRegistry()
    registry.assign(ALICE)
    registry.reset()
    assert registry.count == 0
    assert registry.assign(BOB).speaker_id == "Speaker_01"


@pytest.mark.parametrize("kwargs", [
    {"match_threshold": 1.5},
    {"max_speakers": 0},
    {"momentum": 1.0},
    {"momentum": -0.1},
])
def test_nonsense_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SpeakerRegistry(**kwargs)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def make_identifier(*embeddings: np.ndarray, **kwargs) -> SpeakerIdentifier:
    defaults = dict(embedder=StubEmbedder(*embeddings),
                    registry=SpeakerRegistry(match_threshold=0.5),
                    min_duration_ms=600)
    defaults.update(kwargs)
    return SpeakerIdentifier(**defaults)


def test_a_long_enough_utterance_is_identified():
    identifier = make_identifier(ALICE)
    assignment = identifier.identify(audio(2000))
    assert assignment.speaker_id == "Speaker_01"
    assert identifier.stats.identified == 1


def test_a_short_utterance_is_labelled_unknown_rather_than_guessed():
    """Short interjections usually come from whoever is listening."""
    stub = StubEmbedder(ALICE)
    identifier = make_identifier(embedder=stub)
    assignment = identifier.identify(audio(300))
    assert assignment.speaker_id == SPEAKER_UNKNOWN
    assert "too short" in assignment.reason
    assert stub.calls == 0, "the model should not even be asked"


def test_a_short_utterance_is_not_credited_to_the_previous_speaker():
    identifier = make_identifier(ALICE)
    identifier.identify(audio(2000))
    assert identifier.identify(audio(300)).speaker_id == SPEAKER_UNKNOWN


def test_the_short_cutoff_is_configurable():
    identifier = make_identifier(ALICE, min_duration_ms=100)
    assert identifier.identify(audio(300)).speaker_id == "Speaker_01"


def test_a_broken_voiceprint_is_unknown_rather_than_a_wrong_name():
    identifier = make_identifier(voice(np.nan, 0.0, 0.0))
    assignment = identifier.identify(audio(2000))
    assert assignment.speaker_id == SPEAKER_UNKNOWN
    assert "no usable voiceprint" in assignment.reason


def test_an_empty_voiceprint_is_unknown():
    identifier = make_identifier(np.array([], dtype=np.float64))
    assert identifier.identify(audio(2000)).speaker_id == SPEAKER_UNKNOWN


def test_a_two_speaker_conversation_is_counted_per_speaker():
    identifier = make_identifier(ALICE, BOB, ALICE_AGAIN, BOB)
    for _ in range(4):
        identifier.identify(audio(2000))
    assert identifier.stats.per_speaker == {"Speaker_01": 2, "Speaker_02": 2}
    assert identifier.stats.new_speakers == 2
    assert identifier.stats.unknown == 0


def test_unknowns_are_counted_separately():
    identifier = make_identifier(ALICE)
    identifier.identify(audio(2000))
    identifier.identify(audio(100))
    assert identifier.stats.identified == 1
    assert identifier.stats.unknown == 1
    assert identifier.stats.per_speaker[SPEAKER_UNKNOWN] == 1


def test_reset_starts_the_next_meeting_from_nobody():
    identifier = make_identifier(ALICE)
    identifier.identify(audio(2000))
    identifier.reset()
    assert identifier.stats.seen == 0
    assert identifier.registry.count == 0
