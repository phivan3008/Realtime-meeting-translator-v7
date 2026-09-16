"""Unit tests for the streaming side of the ASR: agreement, commits, the
final tail, and how text is put back together.

The decoder is scripted with timestamped words, so every case here is a
situation read off a real meeting log, replayed without a model.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    ASR_BEAM_SIZE_FINAL,
    ASR_BEAM_SIZE_PARTIAL,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.asr import (  # noqa: E402
    Piece,
    Transcriber,
    Word,
    join_texts,
    render_words,
    spread_words,
)


def audio(seconds: float) -> bytes:
    return bytes(int(seconds * SAMPLE_RATE) * SAMPLE_WIDTH)


def words(*spec) -> tuple[Word, ...]:
    """``("hôm", 0.0, 0.3), ...`` -> Words with Whisper's leading space."""
    return tuple(Word(text=text, start=start, end=end)
                 for text, start, end in spec)


def piece(*spec, text: str = "") -> Piece:
    made = words(*spec)
    return Piece(text=text or "".join(w.text for w in made),
                 avg_logprob=-0.2, no_speech_prob=0.02,
                 compression_ratio=1.4,
                 start=made[0].start if made else 0.0,
                 end=made[-1].end if made else 0.0, words=made)


class ScriptedDecoder:
    """Answers each call from a script of word lists, times relative to the
    audio it is given - which is what faster-whisper does."""

    def __init__(self, *rounds, lang: str = "vi"):
        self.rounds = list(rounds)
        self.lang = lang
        self.calls: list[dict] = []

    def decode(self, samples, lang_code, beam_size, prompt=None):
        self.calls.append({"seconds": samples.size / SAMPLE_RATE,
                           "lang_code": lang_code, "beam_size": beam_size,
                           "prompt": prompt})
        index = min(len(self.calls) - 1, len(self.rounds) - 1)
        return list(self.rounds[index]), (lang_code or self.lang)


def make(*rounds, **kwargs) -> tuple[Transcriber, ScriptedDecoder]:
    decoder = ScriptedDecoder(*rounds, lang=kwargs.pop("lang", "vi"))
    return Transcriber(decoder=decoder, prompt=kwargs.pop("prompt", ""),
                       **kwargs), decoder


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_japanese_is_rendered_without_spaces():
    """The log that found this: この 多 数 ク は 今 朝、 - 88% of one run's
    Japanese sentences arrived with a space between every character."""
    japanese = words(("この", 0.0, 0.2), ("多", 0.2, 0.3), ("数", 0.3, 0.4),
                     ("は", 0.4, 0.5), ("今朝", 0.5, 0.8), ("、", 0.8, 0.8))
    assert render_words(japanese) == "この多数は今朝、"


def test_vietnamese_keeps_whispers_own_spaces():
    vietnamese = words((" Hôm", 0.0, 0.2), (" nay", 0.2, 0.4),
                       (" thì", 0.4, 0.6), (" bác.", 0.6, 0.8))
    assert render_words(vietnamese) == "Hôm nay thì bác."


def test_a_latin_term_split_into_tokens_is_put_back_as_written():
    """Whisper splits Japanese on token boundaries, Latin included."""
    mixed = words(("GL", 0.0, 0.1), ("M", 0.1, 0.2), ("5", 0.2, 0.3),
                  (".2", 0.3, 0.4), ("を", 0.4, 0.5))
    assert render_words(mixed) == "GLM5.2を"


def test_space_before_punctuation_is_removed():
    assert render_words(words((" xin", 0, 1), (" chào", 1, 2), (" .", 2, 2))) \
        == "xin chào."


@pytest.mark.parametrize("left, right, joined", [
    ("Hôm nay", "thì bác", "Hôm nay thì bác"),
    ("今日は", "いい天気", "今日はいい天気"),
    ("GLM5.2を", "使います", "GLM5.2を使います"),
    ("Xin chào", "", "Xin chào"),
    ("", "はい", "はい"),
    ("review", "を", "reviewを"),
])
def test_two_decodes_are_joined_only_where_the_script_wants_a_space(
        left, right, joined):
    assert join_texts(left, right) == joined


def test_words_are_spread_over_a_piece_without_timestamps():
    spread = spread_words(Piece(" one two", -0.2, 0.0, 1.0, 1.0, 2.0),
                          fallback_end=5.0)
    assert [w.text for w in spread] == [" one", " two"]
    assert (spread[0].start, spread[1].end) == (1.0, 2.0)


def test_a_japanese_piece_without_timestamps_gets_no_leading_space():
    spread = spread_words(Piece("はい", -0.2, 0.0, 1.0), fallback_end=0.5)
    assert [w.text for w in spread] == ["はい"]
    assert spread[0].end == 0.5


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
SENTENCE = ((" Hôm", 0.0, 0.3), (" nay", 0.3, 0.6), (" thì", 0.6, 0.9),
            (" bác", 0.9, 1.2), (" nói", 1.2, 1.5))


def test_one_decode_commits_nothing():
    transcriber, _ = make([piece(*SENTENCE)])
    events = transcriber.process_partial(audio(3.0), utterance_id="u",
                                         window_start_seconds=0.0)
    assert [e.kind for e in events] == ["partial"]
    assert events[-1].committed_text == ""
    assert events[-1].partial_text == "Hôm nay thì bác nói"


def test_two_agreeing_decodes_commit_what_is_clear_of_the_edge():
    transcriber, _ = make([piece(*SENTENCE)])
    for _ in range(2):
        events = transcriber.process_partial(audio(2.0), utterance_id="u",
                                             window_start_seconds=0.0)
    assert [e.kind for e in events] == ["committed", "partial"]
    # Everything ending more than a second before the newest audio.
    assert events[0].committed_text == "Hôm nay thì"
    assert events[-1].partial_text == "bác nói"
    assert events[-1].running_text == "Hôm nay thì bác nói"


def test_decodes_that_disagree_commit_nothing():
    other = ((" Hôm", 0.0, 0.3), (" mai", 0.3, 0.6), (" thì", 0.6, 0.9))
    transcriber, _ = make([piece(*SENTENCE)], [piece(*other)])
    for _ in range(2):
        events = transcriber.process_partial(audio(2.0), utterance_id="u",
                                             window_start_seconds=0.0)
    # "Hôm" agrees; "mai" does not, and nothing after a disagreement commits.
    assert events[-1].committed_text == "Hôm"


def test_a_running_text_never_shows_a_committed_word_twice():
    """The same word, placed a little later by the next decode, ended past
    the committed boundary and was shown again."""
    later = ((" Hôm", 0.0, 0.3), (" nay", 0.3, 0.6), (" thì", 0.62, 0.95),
             (" bác", 0.95, 1.2), (" nói", 1.2, 1.5))
    transcriber, _ = make([piece(*SENTENCE)], [piece(*SENTENCE)],
                          [piece(*later)])
    for _ in range(3):
        events = transcriber.process_partial(audio(2.0), utterance_id="u",
                                             window_start_seconds=0.0)
    assert events[-1].running_text == "Hôm nay thì bác nói"


def test_a_sliding_window_is_placed_on_the_utterance_timeline():
    """A window that starts half a second in reports its words half a second
    in, so they line up with the earlier, longer windows."""
    transcriber, _ = make([piece(*SENTENCE)],
                          [piece((" thì", 0.6 - 0.5, 0.9 - 0.5),
                                 (" bác", 0.9 - 0.5, 1.2 - 0.5))])
    transcriber.process_partial(audio(2.5), utterance_id="u",
                                window_start_seconds=0.0)
    events = transcriber.process_partial(audio(2.0), utterance_id="u",
                                         window_start_seconds=0.5)
    assert "thì" in events[-1].committed_text


# ---------------------------------------------------------------------------
# The final tail
# ---------------------------------------------------------------------------
def committed_twice(transcriber, utterance="u"):
    for _ in range(2):
        transcriber.process_partial(audio(2.0), utterance_id=utterance,
                                    window_start_seconds=0.0)


def test_the_final_is_decoded_with_a_beam_and_the_prompt():
    """Greedy decoding doubled the repeated words on a real meeting."""
    transcriber, decoder = make([piece(*SENTENCE)], prompt="solution")
    committed_twice(transcriber)
    transcriber.finish_utterance(audio(2.0), utterance_id="u")
    assert [c["beam_size"] for c in decoder.calls] == [
        ASR_BEAM_SIZE_PARTIAL, ASR_BEAM_SIZE_PARTIAL, ASR_BEAM_SIZE_FINAL]
    assert [c["prompt"] for c in decoder.calls] == [None, None, "solution"]


def test_the_final_decodes_only_the_tail_with_some_context():
    transcriber, decoder = make([piece(*SENTENCE)])
    committed_twice(transcriber)
    transcriber.finish_utterance(audio(2.0), utterance_id="u")
    # committed_end 0.9, minus 1.2 s of context, is the start: all of it.
    assert decoder.calls[-1]["seconds"] == pytest.approx(2.0)


def test_the_last_committed_word_is_not_repeated_by_the_final():
    """'AMD AMD', 'đồ đồ', '朝、 朝、': the tail decode's copy of the last
    committed word landed just past the boundary and was kept."""
    tail = ((" thì", 0.7, 0.95), (" bác", 0.95, 1.2), (" nói", 1.2, 1.5))
    transcriber, decoder = make([piece(*SENTENCE)], [piece(*SENTENCE)],
                                [piece(*tail)])
    committed_twice(transcriber)
    final = transcriber.finish_utterance(audio(2.0), utterance_id="u")
    assert final.text == "Hôm nay thì bác nói"


def test_a_repeat_the_final_places_in_the_wrong_spot_is_still_one_word():
    tail = ((" thì", 1.0, 1.2), (" bác", 1.2, 1.4))
    transcriber, _ = make([piece(*SENTENCE)], [piece(*SENTENCE)],
                          [piece(*tail)])
    committed_twice(transcriber)
    final = transcriber.finish_utterance(audio(2.0), utterance_id="u")
    assert final.text.count("thì") == 1


def test_a_repeat_straddling_the_join_is_removed_there():
    """Everything committed, and the tail decode's copy of the last word
    lands mostly past the boundary."""
    tail = ((" nói", 1.45, 1.7), (" nhé", 1.7, 1.9))
    transcriber, _ = make([piece(*SENTENCE)], [piece(*SENTENCE)],
                          [piece(*tail)])
    for _ in range(2):
        transcriber.process_partial(audio(3.0), utterance_id="u",
                                    window_start_seconds=0.0)
    final = transcriber.finish_utterance(audio(3.0), utterance_id="u")
    assert final.text == "Hôm nay thì bác nói nhé"
    assert transcriber.stats.duplicates_removed == 1


def test_a_japanese_final_joins_its_committed_text_without_a_space():
    committed = (("今日", 0.0, 0.3), ("は", 0.3, 0.5), ("会議", 0.5, 0.9),
                 ("です", 0.9, 1.3))
    tail = (("です", 0.9, 1.3), ("ね", 1.3, 1.5))
    transcriber, _ = make([piece(*committed)], [piece(*committed)],
                          [piece(*tail)], lang="ja")
    committed_twice(transcriber)
    final = transcriber.finish_utterance(audio(2.0), utterance_id="u",
                                         lang_code="ja")
    assert final.text == "今日は会議ですね"


def test_words_invented_over_the_hangover_are_dropped():
    """The VAD forwards half a second of silence after the last word, and
    Whisper does not answer silence with nothing: 'Không, biết thôi đẹp.'"""
    spoken = ((" thì", 0.0, 0.3), (" bạn", 0.3, 0.6), (" fix", 0.6, 0.9))
    invented = spoken + ((" Không", 1.6, 1.8), (" biết", 1.8, 2.0))
    transcriber, decoder = make([piece(*invented)])
    final = transcriber.finish_utterance(audio(2.1), utterance_id="u",
                                         speech_end_seconds=1.0)
    assert final.text == "thì bạn fix"
    # And the silence past the post-roll never reached the model.
    assert decoder.calls[-1]["seconds"] == pytest.approx(1.2)


def test_without_a_speech_end_the_whole_utterance_is_decoded():
    transcriber, decoder = make([piece(*SENTENCE)])
    transcriber.finish_utterance(audio(2.0), utterance_id="u")
    assert decoder.calls[-1]["seconds"] == pytest.approx(2.0)


def test_the_state_is_gone_after_the_final():
    transcriber, _ = make([piece(*SENTENCE)])
    committed_twice(transcriber)
    assert transcriber.has_committed("u")
    transcriber.finish_utterance(audio(2.0), utterance_id="u")
    assert not transcriber.has_committed("u")


def test_a_cancelled_utterance_leaves_nothing_behind():
    transcriber, _ = make([piece(*SENTENCE)])
    committed_twice(transcriber)
    transcriber.cancel_utterance("u")
    assert not transcriber.has_committed("u")


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
def test_the_final_keeps_the_language_its_words_were_committed_in():
    """A sentence whose language disagrees with its running text was
    measured three to four times as likely to be unrelated to what was said.
    The committed words are in the running text's language already."""
    transcriber, decoder = make([piece(*SENTENCE)])
    for _ in range(2):
        transcriber.process_partial(audio(2.0), utterance_id="u",
                                    window_start_seconds=0.0, lang_code="vi")
    final = transcriber.finish_utterance(audio(2.0), utterance_id="u",
                                         lang_code="ja")
    assert final.lang_code == "vi"
    assert final.overruled_language == "ja"
    assert decoder.calls[-1]["lang_code"] == "vi"


def test_with_nothing_committed_the_final_takes_the_language_it_is_given():
    transcriber, decoder = make([piece(*SENTENCE)])
    transcriber.process_partial(audio(2.0), utterance_id="u",
                                window_start_seconds=0.0, lang_code="vi")
    final = transcriber.finish_utterance(audio(2.0), utterance_id="u",
                                         lang_code="ja")
    assert final.lang_code == "ja"
    assert final.overruled_language == ""


def test_a_short_scrap_is_judged_on_no_speech_prob_alone_while_streaming():
    scrap = Piece(" Cảm ơn nhé", -0.3, 0.86, 1.2, 0.0, 0.4,
                  words((" Cảm", 0.0, 0.1), (" ơn", 0.1, 0.2),
                        (" nhé", 0.2, 0.4)))
    transcriber, _ = make([scrap])
    events = transcriber.process_partial(audio(0.4), utterance_id="u",
                                         window_start_seconds=0.0)
    assert events[-1].running_text == ""
    final = transcriber.finish_utterance(audio(0.5), utterance_id="u")
    assert final.text == ""
