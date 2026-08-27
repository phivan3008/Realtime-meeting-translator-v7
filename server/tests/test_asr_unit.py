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
from server.wordlists import (
    Hallucinations,
    normalise_exact,
    normalise_spaced,
)


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


def test_a_segment_whisper_thinks_is_silence_and_is_unsure_of_is_dropped():
    transcript = make([Piece(" mm hm", -1.4, 0.95, 1.4)]).transcribe(audio())
    assert transcript.text == ""
    assert transcript.dropped[0][1] == "no speech"


def test_confident_speech_survives_a_high_no_speech_score():
    """A real sentence, lost on a real meeting: 6.8 s of one man talking,
    refused as silence. Whisper's own rule needs both signals, and says so:
    "don't skip if the logprob is high enough, despite the no_speech_prob".
    Reading no_speech_prob alone kills speech and keeps inventions, because
    Whisper writes its sign-offs more confidently than it writes real
    speech - which is what server/data/ is for."""
    real = Piece(" 2011 thì mình đang lấy bởi vì là cái cả AMD mà bắt cung cấp",
                 -0.35, 0.86, 1.6)
    transcript = make([real]).transcribe(audio())
    assert transcript.dropped == ()
    assert "AMD" in transcript.text


def test_a_confident_sign_off_over_silence_is_still_refused():
    """It now falls through to the word list, which is the layer that can
    actually tell it apart from speech."""
    invented = Piece(" Thank you for watching!", -0.3, 0.95, 1.4)
    transcript = make([invented]).transcribe(audio())
    assert transcript.text == ""
    assert transcript.dropped[0][1] == "known hallucination"


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
        Piece(" Subscribe to my channel", -1.5, 0.99, 1.3),
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
        [Piece(" a", -1.2, 0.99, 1.5)],
        [Piece(" b", -3.0, 0.1, 1.5)],
    )
    transcriber.transcribe(audio())
    transcriber.transcribe(audio())
    assert transcriber.stats.dropped_pieces == 2
    assert transcriber.stats.dropped_reasons == {"no speech": 1,
                                                 "low confidence": 1}


def test_stats_count_transcripts_that_came_back_empty():
    transcriber = make([Piece(" x", -1.2, 0.99, 1.5)])
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


# ---------------------------------------------------------------------------
# Sign-offs Whisper invents out of near-silence
#
# The 60 s end-to-end run put "Cảm ơn các bạn đã theo dõi và hẹn gặp lại." on
# screen as running text over a meeting about task tables. It passed every
# check in _refuse, because Whisper writes its sign-offs with a *lower*
# no_speech_prob and a *higher* avg_logprob than it writes real speech.
# ---------------------------------------------------------------------------
SIGN_OFF_VI = ("C\u1ea3m \u01a1n c\u00e1c b\u1ea1n \u0111\u00e3 theo d\u00f5i "
               "v\u00e0 h\u1eb9n g\u1eb7p l\u1ea1i.")
SIGN_OFF_JA = "\u3054\u8996\u8074\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\u3002"


def confident(text: str) -> Piece:
    """What Whisper's numbers look like when it invents a sign-off: better
    than its numbers on real speech."""
    return Piece(text=f" {text}", avg_logprob=-0.15, no_speech_prob=0.02,
                 compression_ratio=1.5)


def test_the_fixtures_survived_being_written_to_disk():
    assert SIGN_OFF_VI.startswith("C\u1ea3m \u01a1n")
    assert SIGN_OFF_JA.startswith("\u3054\u8996\u8074")


def test_the_statistical_guards_would_have_kept_it():
    """The reason a content rule had to exist at all: with the list emptied,
    every other guard waves this line straight through."""
    transcript = make([confident(SIGN_OFF_VI)],
                      hallucinations=Hallucinations(exact=[], patterns=(), keep=[])).transcribe(audio())
    assert transcript.text == SIGN_OFF_VI
    assert transcript.dropped == ()


def test_the_list_is_not_empty():
    """An empty list would make every test above pass for the wrong reason."""
    assert len(Hallucinations()) > 0


def test_the_vietnamese_sign_off_is_refused():
    transcript = make([confident(SIGN_OFF_VI)]).transcribe(audio())
    assert transcript.text == ""
    assert [reason for _p, reason in transcript.dropped] == [
        "known hallucination"
    ]


def test_the_japanese_sign_off_is_refused():
    transcript = make([confident(SIGN_OFF_JA)]).transcribe(audio())
    assert transcript.text == ""


def test_the_english_answer_to_silence_is_refused():
    transcript = make([confident("you")]).transcribe(audio())
    assert transcript.text == ""


def test_punctuation_and_case_do_not_let_it_through():
    """Whisper punctuates its own inventions inconsistently."""
    for variant in (SIGN_OFF_VI, SIGN_OFF_VI.rstrip("."),
                    SIGN_OFF_VI.upper(), f"  {SIGN_OFF_VI}  "):
        transcript = make([confident(variant)]).transcribe(audio())
        assert transcript.text == "", variant


def test_a_real_sentence_containing_the_phrase_is_kept():
    """Matched in full, so a meeting may still thank people for watching."""
    real = ("C\u1ea3m \u01a1n c\u00e1c b\u1ea1n \u0111\u00e3 theo d\u00f5i "
            "b\u00e1o c\u00e1o n\u00e0y.")
    transcript = make([confident(real)]).transcribe(audio())
    assert transcript.text == real


def test_the_confirmed_goodbye_is_refused():
    """Committed as a sentence at 55.3 s of the second end-to-end run. The
    recording was played back over 54-56 s and nobody spoke it."""
    goodbye = "Chào tạm biệt."
    assert make([confident(goodbye)]).transcribe(audio()).text == ""


@pytest.mark.parametrize("said", [
    "Chào tạm biệt nhé.",
    "Thôi chào tạm biệt mọi người.",
    "Ok chào tạm biệt.",
    "Tạm biệt.",
])
def test_a_goodbye_with_anything_attached_survives(said):
    """Whole-segment matching is the only thing keeping that entry narrow.

    Unlike the video sign-offs, this one blocks an ordinary Vietnamese
    sentence, and these are the sentences that cost must not reach.
    """
    assert make([confident(said)]).transcribe(audio()).text == said


def test_a_question_containing_you_is_kept():
    assert make([confident("Are you there?")]).transcribe(audio()).text == \
        "Are you there?"


def test_the_refusal_is_counted_under_its_own_reason():
    transcriber = make([confident(SIGN_OFF_VI)])
    transcriber.transcribe(audio())
    assert transcriber.stats.dropped_reasons == {"known hallucination": 1}


def test_the_refused_text_reaches_the_log(caplog):
    """A guard that hides what it refused turns one bug into two."""
    import logging
    with caplog.at_level(logging.INFO, logger="server.pipeline.asr"):
        make([confident(SIGN_OFF_VI)]).transcribe(audio())
    assert "known hallucination" in caplog.text
    assert "theo d\u00f5i" in caplog.text


def test_normalising_keeps_vietnamese_letters_apart():
    """Diacritics are letters here, not punctuation: folding them would run
    different words together."""
    assert normalise_exact("t\u1eaft") != normalise_exact("tab")
    assert normalise_exact("\u0111\u00f3") != normalise_exact("do")


def test_the_youtube_subscribe_pitch_is_refused():
    """Running text at 107.3 s of the eighth run, over a meeting about task
    tables. A named channel's subscribe pitch is not something a meeting
    says."""
    pitch = ("H\u00e3y subscribe cho k\u00eanh Ghi\u1ec1n M\u00ec G\u00f5 "
             "\u0110\u1ec3 kh\u00f4ng b\u1ecf l\u1ee1 nh\u1eefng video "
             "h\u1ea5p d\u1eabn")
    assert make([confident(pitch)]).transcribe(audio()).text == ""


def test_a_meeting_asking_people_to_subscribe_survives():
    """Whole-segment matching again: only that exact pitch is blocked."""
    real = ("H\u00e3y subscribe cho k\u00eanh c\u1ee7a b\u00ean m\u00ecnh "
            "nh\u00e9")
    assert make([confident(real)]).transcribe(audio()).text == real


def test_the_second_youtube_pitch_is_refused():
    """Committed as a sentence at 7.6 s of the ninth run and translated into
    Japanese before anybody noticed."""
    pitch = ("H\u00e3y \u0111\u0103ng k\u00fd k\u00eanh \u0111\u1ec3 "
             "\u1ee7ng h\u1ed9 k\u00eanh c\u1ee7a m\u00ecnh nh\u00e9.")
    assert make([confident(pitch)]).transcribe(audio()).text == ""


def test_a_meeting_asking_people_to_subscribe_to_something_else_survives():
    real = ("H\u00e3y \u0111\u0103ng k\u00fd k\u00eanh Teams cho d\u1ef1 "
            "\u00e1n n\u00e0y")
    assert make([confident(real)]).transcribe(audio()).text == real


def test_the_same_pitch_with_a_different_channel_is_caught():
    """This is what the whole-line list could not do, and the reason the
    pattern rules exist. Two runs after the Ghiền Mì Gõ line was listed, the
    identical sentence arrived naming La La School and was translated into
    Japanese and shown."""
    for channel in ("Ghiền Mì Gõ", "La La School",
                    "anything at all"):
        pitch = (f"Hãy subscribe cho kênh {channel} "
                 "Để không bỏ lỡ những "
                 "video hấp dẫn")
        assert make([confident(pitch)]).transcribe(audio()).text == "", channel


def test_the_subscribe_pitch_is_caught_with_or_without_a_channel():
    for line in ("Hãy đăng ký kênh để "
                 "ủng hộ kênh của mình nhé.",
                 "Hãy đăng ký kênh ABC XYZ "
                 "để ủng hộ kênh của "
                 "mình"):
        assert make([confident(line)]).transcribe(audio()).text == "", line


@pytest.mark.parametrize("said", [
    # A meeting may well talk about channels. The pattern needs the rest of
    # the pitch behind it, not just the opening.
    "Hãy đăng ký kênh Teams cho dự án này",
    "Hãy subscribe cho kênh nội bộ của team mình",
    "Chúng ta hãy đăng ký kênh Slack trước",
])
def test_a_meeting_talking_about_channels_survives(said):
    assert make([confident(said)]).transcribe(audio()).text == said


def test_a_pattern_must_match_the_whole_line():
    """Anchored, so a real sentence that happens to contain the pitch is not
    deleted along with it."""
    longer = ("Khách hàng bảo hãy subscribe cho "
              "kênh La La School Để không bỏ "
              "lỡ những video hấp dẫn, nhưng "
              "mình không đồng ý.")
    assert make([confident(longer)]).transcribe(audio()).text == longer


def test_the_patterns_are_not_empty():
    """An empty tuple would make every test above pass for nothing."""
    assert Hallucinations().patterns


def test_the_patterns_can_be_replaced_for_a_test():
    """With them emptied, the reworded pitch sails through - which is what
    the whole-line list did on its own."""
    pitch = ("Hãy subscribe cho kênh La La School "
             "Để không bỏ lỡ những "
             "video hấp dẫn")
    kept = make([confident(pitch)], hallucinations=Hallucinations(patterns=(), keep=[]))
    assert kept.transcribe(audio()).text == pitch


def test_pattern_normalising_keeps_words_apart():
    """The exact rules strip spacing; the patterns read as sentences and
    cannot."""
    assert normalise_spaced("Hãy  subscribe   cho kênh!") == \
        "hãy subscribe cho kênh"


@pytest.mark.parametrize("line", [
    "C\u00e1c b\u1ea1n c\u00f3 th\u1ec3 nh\u1edb like, share v\u00e0 "
    "\u0111\u0103ng k\u00fd k\u00eanh \u0111\u1ec3 \u1ee7ng h\u1ed9 "
    "k\u00eanh c\u1ee7a m\u00ecnh nh\u00e9.",
    "C\u00e1c b\u1ea1n c\u00f3 th\u1ec3 nh\u1edb like v\u00e0 share video "
    "n\u00e0y \u0111\u1ec3 \u1ee7ng h\u1ed9 k\u00eanh c\u1ee7a m\u00ecnh",
    "C\u00e1c b\u1ea1n c\u00f3 th\u1ec3 nh\u1edb \u0111\u0103ng k\u00fd "
    "k\u00eanh \u0111\u1ec3 \u1ee7ng h\u1ed9 k\u00eanh c\u1ee7a m\u00ecnh "
    "nh\u00e9.",
])
def test_the_like_and_share_pitch_is_caught_in_all_its_variants(line):
    """Three variants of one shape in a single ten-minute run. The middle
    clause is the hole this time, not a channel name."""
    assert make([confident(line)]).transcribe(audio()).text == ""


@pytest.mark.parametrize("said", [
    "C\u00e1c b\u1ea1n c\u00f3 th\u1ec3 nh\u1edb g\u1eedi t\u00e0i li\u1ec7u "
    "cho t\u00f4i",
    "C\u00e1c b\u1ea1n c\u00f3 th\u1ec3 nh\u1edb m\u1edf k\u00eanh Teams "
    "l\u00ean kh\u00f4ng",
])
def test_a_meeting_reminding_people_of_something_survives(said):
    assert make([confident(said)]).transcribe(audio()).text == said


def test_these_were_only_caught_by_luck_before():
    """Every one of them was refused by no_speech_prob on that run. The same
    lines have arrived confident on other runs, and a statistical guard that
    happens to fire is not a policy."""
    line = ("C\u00e1c b\u1ea1n c\u00f3 th\u1ec3 nh\u1edb \u0111\u0103ng "
            "k\u00fd k\u00eanh \u0111\u1ec3 \u1ee7ng h\u1ed9 k\u00eanh "
            "c\u1ee7a m\u00ecnh nh\u00e9.")
    kept = make([confident(line)], hallucinations=Hallucinations(patterns=(), keep=[]))
    assert kept.transcribe(audio()).text == line


# ---------------------------------------------------------------------------
# Short audio, confident nonsense
# ---------------------------------------------------------------------------
def short_audio(ms: float = 400) -> bytes:
    return np.zeros(int(ms * SAMPLE_RATE / 1000.0), dtype="<i2").tobytes()


def test_a_confident_invention_over_a_scrap_of_audio_is_refused():
    """Confirmed on a real meeting: every "Cảm ơn..." reaching a committed
    sentence was invented, and they all came from scraps too short for the
    speaker model or the LID to answer at all. Whisper answers anyway, and
    confidently - which is how the relaxed no-speech rule lets them past."""
    invented = Piece(" Cảm ơn các bạn.", -0.31, 0.86, 1.4)
    transcript = make([invented]).transcribe(short_audio(400))
    assert transcript.text == ""
    assert transcript.dropped[0][1] == "no speech"


def test_the_long_sentence_that_rule_was_built_for_still_survives():
    """The 6.8 s of one man talking that reading no_speech_prob alone threw
    away. Duration is what separates it from the inventions."""
    real = Piece(" 2011 thì mình đang lấy bởi vì là cái cả AMD mà bắt cung cấp",
                 -0.35, 0.86, 1.6)
    transcript = make([real]).transcribe(short_audio(6_800))
    assert transcript.dropped == ()
    assert "AMD" in transcript.text


def test_a_scrap_that_whisper_is_sure_is_speech_is_kept():
    """Short is not the same as invented. A one-word answer is a real thing
    for somebody to say."""
    real = Piece(" Vâng.", -0.30, 0.05, 1.2)
    transcript = make([real]).transcribe(short_audio(400))
    assert transcript.dropped == ()
