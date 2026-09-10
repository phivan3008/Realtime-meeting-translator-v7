"""Unit tests for replaying the ASR guards over recorded scores.

The point of these is parity. The simulation exists so a threshold question
can be answered without a GPU, and it is worth nothing if it drifts away from
what ``Transcriber._refuse`` actually does. So the ``current`` rule is checked
against the live one on every kind of segment, and the ``whisper`` rule is
checked to differ on exactly one of them.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.analysis import guards  # noqa: E402
from server.pipeline.asr import Piece  # noqa: E402


@pytest.fixture
def transcriber():
    return guards.make_transcriber()


def piece(text=" hôm nay chúng ta họp", avg_logprob=-0.2, no_speech_prob=0.05,
          compression_ratio=1.6) -> Piece:
    return Piece(text=text, avg_logprob=avg_logprob,
                 no_speech_prob=no_speech_prob,
                 compression_ratio=compression_ratio)


#: One of each kind of segment the guards have an opinion about.
CASES = {
    "ordinary speech": piece(),
    "empty": piece(text="   "),
    "silence, decoded badly": piece(no_speech_prob=0.95, avg_logprob=-1.4),
    "silence, decoded well": piece(no_speech_prob=0.95, avg_logprob=-0.2),
    "low confidence": piece(avg_logprob=-1.5),
    "looping": piece(compression_ratio=3.1),
    "known invention": piece(text=" Cảm ơn các bạn đã theo dõi."),
    "invention, high scores": piece(text=" Cảm ơn các bạn đã theo dõi.",
                                    avg_logprob=-0.1, no_speech_prob=0.01),
}


class TestParity:
    @pytest.mark.parametrize("name", sorted(CASES))
    def test_the_current_rule_is_the_live_one(self, transcriber, name):
        """Not a reimplementation - it defers to the Transcriber itself."""
        assert (guards.refuse(transcriber, CASES[name], "current")
                == transcriber._refuse(CASES[name]))

    @pytest.mark.parametrize("name", sorted(set(CASES) - {"silence, decoded well"}))
    def test_the_whisper_rule_agrees_everywhere_else(self, transcriber, name):
        assert (guards.refuse(transcriber, CASES[name], "whisper")
                == guards.refuse(transcriber, CASES[name], "current"))

    def test_the_one_case_they_differ_on(self, transcriber):
        """A confident decode of a segment Whisper doubted was speech.

        Today it is thrown away. Whisper's own rule keeps it, and that is the
        whole question this simulation exists to answer.
        """
        confident_silence = CASES["silence, decoded well"]
        assert guards.refuse(transcriber, confident_silence, "current") == "no speech"
        assert guards.refuse(transcriber, confident_silence, "whisper") is None

    def test_an_unknown_rule_is_refused(self, transcriber):
        with pytest.raises(ValueError, match="unknown guard rule"):
            guards.refuse(transcriber, piece(), "whatever")


class TestOrdering:
    def test_an_invention_is_still_caught_under_the_whisper_rule(self,
                                                                 transcriber):
        """Relaxing no_speech must not open a hole for the word lists.

        Whisper writes its sign-offs with a *higher* avg_logprob than it
        writes real speech, so this is precisely the segment that the relaxed
        rule would otherwise wave through.
        """
        assert guards.refuse(transcriber, CASES["invention, high scores"],
                             "whisper") == "known hallucination"

    def test_a_loop_is_still_caught_under_the_whisper_rule(self, transcriber):
        looping_silence = piece(no_speech_prob=0.95, avg_logprob=-0.2,
                                compression_ratio=3.1)
        assert guards.refuse(transcriber, looping_silence,
                             "whisper") == "repetition"

    def test_the_english_sign_off_is_on_no_list_and_this_is_the_risk(
            self, transcriber):
        """The cost of the rule change, stated as a test.

        "Thank you for watching!" is what Whisper answers silence with in
        English, and ``ASR_HALLUCINATIONS`` does not have it - the list is
        Vietnamese and Japanese sign-offs, because those are what this project
        has seen reach a screen. Today the line is refused by
        ``no_speech_prob`` alone. Under Whisper's rule it is kept, because
        Whisper writes it confidently.

        So relaxing the rule without adding the line is a trade, not a fix,
        and this test fails the moment somebody adds it - which is the point
        at which the trade is gone.
        """
        english = piece(text=" Thank you for watching!", avg_logprob=-0.1,
                        no_speech_prob=0.95)
        assert guards.refuse(transcriber, english, "current") == "no speech"
        assert guards.refuse(transcriber, english, "whisper") is None


class TestDecide:
    def test_the_sentence_is_rebuilt_in_order(self, transcriber):
        pieces = [piece(text=" một"), piece(text=" hai"), piece(text=" ba")]
        assert guards.decide(transcriber, pieces, "current")["text"] == (
            "một hai ba")

    def test_a_refused_segment_leaves_the_others_alone(self, transcriber):
        pieces = [piece(text=" một"),
                  piece(text=" hai", no_speech_prob=0.95, avg_logprob=-1.4),
                  piece(text=" ba")]
        verdict = guards.decide(transcriber, pieces, "current")
        assert verdict["text"] == "một ba"
        assert [reason for _p, reason in verdict["dropped"]] == ["no speech"]

    def test_the_whisper_rule_puts_a_confident_segment_back(self, transcriber):
        pieces = [piece(text=" một"),
                  piece(text=" hai", no_speech_prob=0.95, avg_logprob=-0.2)]
        assert guards.decide(transcriber, pieces, "current")["text"] == "một"
        assert guards.decide(transcriber, pieces, "whisper")["text"] == "một hai"


def run_with(pieces_by_index: dict) -> dict:
    return {
        "wav": "x.wav", "audio_seconds": 60.0,
        "variants": [{"name": "baseline", "beam": 5, "trim_ms": 0.0}],
        "cases": [
            {"index": index, "start_ms": 0.0, "end_ms": 5000.0,
             "reason": "pause", "continues_previous": False, "lang_code": "vi",
             "partial_text": "một hai", "partial_count": 5,
             "extra_audio_ms": 300.0, "drift": {},
             "finals": {"baseline": {"text": "một", "seconds": 0.2,
                                     "audio_ms": 5000.0, "dropped": [],
                                     "pieces": records}}}
            for index, records in pieces_by_index.items()
        ],
    }


def record(text, avg_logprob=-0.2, no_speech_prob=0.05,
           compression_ratio=1.6, verdict="kept") -> dict:
    return {"text": text, "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio, "verdict": verdict}


class TestSimulate:
    def test_a_run_is_replayed_sentence_by_sentence(self):
        run = run_with({0: [record(" một"),
                            record(" hai", no_speech_prob=0.95,
                                   verdict="no speech")]})
        assert guards.simulate(run, "baseline", "current")[0]["text"] == "một"
        assert guards.simulate(run, "baseline", "whisper")[0]["text"] == (
            "một hai")

    def test_a_run_without_scores_is_skipped_rather_than_guessed_at(self):
        run = run_with({0: []})
        assert guards.simulate(run, "baseline", "whisper") == {}
        assert not guards.has_scores(run, "baseline")

    def test_a_run_with_scores_says_so(self):
        assert guards.has_scores(run_with({0: [record(" một")]}), "baseline")

    def test_the_reasons_come_back_with_the_text(self):
        run = run_with({0: [record(" Cảm ơn các bạn đã theo dõi.")]})
        result = guards.simulate(run, "baseline", "whisper")[0]
        assert result["text"] == ""
        assert result["reasons"] == ["known hallucination"]
        assert result["kept"] == 0


class TestNearMiss:
    """The block list matches whole segments, so a trimmed invention walks
    through it. This does not block anything - it says what to read."""

    def test_a_trimmed_sign_off_resembles_the_listed_one(self):
        found = guards.near_miss("Cảm ơn các bạn.")
        assert found is not None
        assert found["listed"] == "Cảm ơn các bạn đã theo dõi."
        assert found["shared_prefix"]

    def test_a_reworded_sign_off_is_caught_by_its_opening(self):
        found = guards.near_miss("Hẹn gặp lại mọi người")
        assert found is not None
        assert found["listed"].startswith("Hẹn gặp lại")

    def test_a_real_meeting_sentence_resembles_nothing(self):
        assert guards.near_miss(
            "Về tác 08 thì hiện tại mình đang thực hiện test 1") is None
        assert guards.near_miss(
            "2011 thì mình đang lên bởi vì là cái cả AMD mà bắt cung cấp") is None

    def test_empty_text_resembles_nothing(self):
        assert guards.near_miss("") is None
        assert guards.near_miss("   ...  ") is None

    def test_an_exact_listed_line_is_its_own_nearest(self):
        found = guards.near_miss("Cảm ơn các bạn đã theo dõi.")
        assert found["distance"] == 0.0

    def test_the_list_it_compares_against_can_be_given(self):
        assert guards.near_miss("hello there", phrases=("hello world",))
        assert guards.near_miss("hello there", phrases=("完全に違う",)) is None
