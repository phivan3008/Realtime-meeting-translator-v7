"""Unit tests for the ASR guards.

Whisper is stubbed, so these run without the model. Whether it transcribes
Vietnamese and Japanese correctly is a question for
``server/tests_real/test_real_asr.py`` on the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (
    ASR_BEAM_SIZE_FINAL,
    ASR_BEAM_SIZE_PARTIAL,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.asr import Piece, Transcriber, pcm_seconds


class StubDecoder:
    """Returns scripted segments and remembers how it was called."""

    def __init__(self, *rounds: list):
        self.rounds = list(rounds) or [[]]
        self.calls: list[dict] = []

    def decode(self, samples, lang_code, beam_size):
        self.calls.append({"samples": samples, "lang_code": lang_code,
                           "beam_size": beam_size})
        pieces = self.rounds[min(len(self.calls) - 1, len(self.rounds) - 1)]
        return list(pieces), "vi"


def good(text: str = " hello there ") -> Piece:
    return Piece(text=text, avg_logprob=-0.2, no_speech_prob=0.05,
                 compression_ratio=1.6)


def audio(ms: float = 2000) -> bytes:
    return bytes(int(ms * SAMPLE_RATE / 1000) * SAMPLE_WIDTH)


def make(*rounds: list, **kwargs) -> Transcriber:
    return Transcriber(decoder=StubDecoder(*rounds), **kwargs)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def test_pcm_seconds_reads_the_audio_contract():
    assert pcm_seconds(audio(2000)) == pytest.approx(2.0)
    assert pcm_seconds(b"") == 0.0


def test_the_transcript_joins_the_segments_it_kept():
    transcript = make([good(" one "), good(" two ")]).transcribe(audio())
    assert transcript.text == "one two"
    assert transcript.has_text is True


def test_audio_is_handed_over_as_floats_in_range():
    stub = StubDecoder([good()])
    Transcriber(decoder=stub).transcribe(
        np.full(SAMPLE_RATE, 16384, dtype="<i2").tobytes()
    )
    samples = stub.calls[0]["samples"]
    assert samples.dtype == np.float32
    assert samples.max() == pytest.approx(0.5, abs=0.01)


def test_empty_audio_is_not_sent_to_the_model():
    stub = StubDecoder([good()])
    transcript = Transcriber(decoder=stub).transcribe(b"")
    assert transcript.text == ""
    assert stub.calls == []


# ---------------------------------------------------------------------------
# Partial against final
# ---------------------------------------------------------------------------
def test_a_partial_is_decoded_greedily_and_a_final_with_a_beam():
    """A partial is replaced within a second; a final is what the viewer keeps."""
    stub = StubDecoder([good()])
    transcriber = Transcriber(decoder=stub)
    transcriber.transcribe(audio(), is_final=False)
    transcriber.transcribe(audio(), is_final=True)
    assert stub.calls[0]["beam_size"] == ASR_BEAM_SIZE_PARTIAL
    assert stub.calls[1]["beam_size"] == ASR_BEAM_SIZE_FINAL
    assert ASR_BEAM_SIZE_PARTIAL < ASR_BEAM_SIZE_FINAL


def test_the_transcript_says_which_mode_produced_it():
    assert make([good()]).transcribe(audio(), is_final=False).is_final is False
    assert make([good()]).transcribe(audio(), is_final=True).is_final is True


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
def test_a_known_language_is_forced_on_the_decoder():
    stub = StubDecoder([good()])
    Transcriber(decoder=stub).transcribe(audio(), lang_code="ja")
    assert stub.calls[0]["lang_code"] == "ja"


def test_an_unknown_language_lets_the_model_detect_and_reports_what_it_found():
    """Which is the whole point of the LID stage returning an empty answer."""
    transcript = make([good()]).transcribe(audio(), lang_code="")
    assert transcript.lang_code == "vi"


def test_a_forced_language_is_kept_in_the_transcript():
    assert make([good()]).transcribe(audio(), lang_code="ja").lang_code == "ja"


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------
def test_clean_speech_is_kept():
    transcript = make([good()]).transcribe(audio())
    assert transcript.dropped == ()
    assert len(transcript.kept) == 1


def test_a_segment_whisper_thinks_is_silence_is_dropped():
    """This is the "Thank you for watching" that appears over quiet audio."""
    invented = Piece(" Thank you for watching!", -0.3, 0.95, 1.4)
    transcript = make([invented]).transcribe(audio())
    assert transcript.text == ""
    assert transcript.dropped[0][1] == "no speech"


def test_a_low_confidence_segment_is_dropped():
    transcript = make([Piece(" mumble", -2.5, 0.1, 1.5)]).transcribe(audio())
    assert transcript.dropped[0][1] == "low confidence"


def test_a_repetition_loop_is_dropped():
    """Whisper filling time with one phrase compresses far better than speech."""
    loop = Piece(" yes yes yes yes yes yes yes yes", -0.4, 0.1, 8.0)
    transcript = make([loop]).transcribe(audio())
    assert transcript.dropped[0][1] == "repetition"


def test_an_empty_segment_is_dropped_without_blaming_the_model():
    transcript = make([Piece("   ", -0.2, 0.05, 1.5)]).transcribe(audio())
    assert transcript.dropped[0][1] == "empty"


def test_the_good_segments_survive_a_bad_neighbour():
    """One invented segment must not take the real sentence down with it."""
    transcript = make([
        good(" the real sentence "),
        Piece(" Subscribe to my channel", -0.3, 0.99, 1.3),
    ]).transcribe(audio())
    assert transcript.text == "the real sentence"
    assert len(transcript.kept) == 1
    assert len(transcript.dropped) == 1


def test_the_guards_are_configurable():
    borderline = Piece(" maybe", -1.5, 0.5, 2.0)
    assert make([borderline], log_prob_threshold=-2.0).transcribe(audio()).has_text
    assert not make([borderline], log_prob_threshold=-1.0).transcribe(audio()).has_text


def test_a_transcript_summarises_itself_for_the_log():
    summary = make([good(" hi ")]).transcribe(audio(), lang_code="vi").summary()
    assert "final" in summary
    assert "[vi]" in summary
    assert "1 kept" in summary


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def test_stats_separate_partials_from_finals():
    transcriber = make([good()])
    transcriber.transcribe(audio(), is_final=False)
    transcriber.transcribe(audio(), is_final=True)
    assert transcriber.stats.partials == 1
    assert transcriber.stats.finals == 1


def test_stats_count_why_segments_were_dropped():
    transcriber = make(
        [Piece(" a", -0.2, 0.99, 1.5)],
        [Piece(" b", -3.0, 0.1, 1.5)],
    )
    transcriber.transcribe(audio())
    transcriber.transcribe(audio())
    assert transcriber.stats.dropped_pieces == 2
    assert transcriber.stats.dropped_reasons == {"no speech": 1,
                                                 "low confidence": 1}


def test_stats_count_transcripts_that_came_back_empty():
    transcriber = make([Piece(" x", -0.2, 0.99, 1.5)])
    transcriber.transcribe(audio())
    assert transcriber.stats.empty == 1


def test_the_realtime_factor_needs_audio_before_it_means_anything():
    transcriber = make([good()])
    assert transcriber.stats.realtime_factor == 0.0
    transcriber.transcribe(audio(2000))
    assert transcriber.stats.audio_seconds == pytest.approx(2.0)


def test_reset_clears_the_counters():
    transcriber = make([good()])
    transcriber.transcribe(audio())
    transcriber.reset()
    assert transcriber.stats.finals == 0
    assert transcriber.stats.dropped_reasons == {}
