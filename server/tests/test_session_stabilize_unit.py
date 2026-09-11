"""Unit tests for checking a committed sentence against its running texts.

The fault being caught: the ASR is forced into one language per utterance,
and forcing the wrong one does not fail - it returns fluent text in a
language nobody spoke. Measured over three real meetings, a sentence whose
language disagrees with its running text is three to four times as likely to
be unrelated to what was said, so that disagreement is the only thing this
policy acts on.

No socket, no model: the decoder is scripted and says whichever language it
is told to.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.protocol import CHUNK_BYTES, Hello  # noqa: E402
from server.net.session import ServerSession  # noqa: E402
from server.pipeline.asr import Piece, Transcriber  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES, VADSegmenter  # noqa: E402


class ScriptedVAD:
    def __init__(self, probabilities):
        self.script = list(probabilities)
        self.calls = 0

    def probability(self, frame: np.ndarray) -> float:
        assert frame.shape[-1] == VAD_FRAME_SAMPLES
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return float(value)

    def reset(self) -> None:
        self.calls = 0


class ScriptedDecoder:
    """Answers by beam, so the running text and the sentence can differ.

    beam 1 is the running text; beam 5 is the committed sentence. A real
    decoder does not work this way - this is how a test says "these two
    readings of the same audio disagreed".
    """

    def __init__(self, partial: str, partial_lang: str,
                 final: str, final_lang: str,
                 retry: str = "", retry_lang: str = ""):
        self.partial, self.partial_lang = partial, partial_lang
        self.final, self.final_lang = final, final_lang
        self.retry, self.retry_lang = retry, retry_lang
        self.calls: list = []

    def decode(self, samples, lang_code: str = "", beam_size: int = 1):
        self.calls.append({"lang_code": lang_code, "beam": beam_size})
        if beam_size == 1:
            text, lang = self.partial, self.partial_lang
        elif lang_code and lang_code == self.retry_lang:
            text, lang = self.retry, self.retry_lang
        else:
            text, lang = self.final, self.final_lang
        if not text:
            return [], lang
        return [Piece(text=text, no_speech_prob=0.01, avg_logprob=-0.2,
                      compression_ratio=1.2)], (lang_code or lang)


def session_with(decoder) -> ServerSession:
    vad = ScriptedVAD([0.9])
    session = ServerSession(segmenter_factory=lambda: VADSegmenter(vad=vad),
                            transcriber=Transcriber(decoder=decoder))
    session.handle_text(Hello(session_id="abc").to_json())
    return session


def chunk(value: int = 1000) -> bytes:
    return np.full(CHUNK_BYTES // 2, value, dtype="<i2").tobytes()


def speak(session, speech: int = 12, silence: int = 0) -> list:
    """Enough audio for several running texts, then a pause to commit."""
    messages = []
    for _ in range(speech):
        messages += session.handle_binary(chunk()).messages
    if silence:
        session.segmenter.vad.script = [0.02]
        session.segmenter.vad.calls = 0
        for _ in range(silence):
            messages += session.handle_binary(chunk(0)).messages
    messages += session.finish().messages
    return [json.loads(message) for message in messages]


def finals(messages) -> list:
    return [message for message in messages if message["type"] == "final"]


def partials(messages) -> list:
    return [message for message in messages if message["type"] == "partial"]


class TestAgreement:
    def test_a_sentence_that_agrees_with_its_running_text_is_untouched(self):
        decoder = ScriptedDecoder(
            partial="hôm nay chúng ta họp", partial_lang="vi",
            final="Hôm nay chúng ta họp về tiến độ.", final_lang="vi")
        session = session_with(decoder)
        committed = finals(speak(session))
        assert committed
        assert committed[0]["transcript"] == "Hôm nay chúng ta họp về tiến độ."
        assert session.stats.language_retries == 0
        assert session.stats.sentences_from_running_text == 0

    def test_drifting_text_alone_is_not_acted_on(self):
        """Nothing measured says the running text is the better of the two.

        It is decoded greedily on four seconds; the sentence gets a beam
        search over the whole utterance. Where they merely word things
        differently, the sentence is the better telling - which is the whole
        reason it is decoded again.
        """
        decoder = ScriptedDecoder(
            partial="về solution và cái lý do", partial_lang="vi",
            final="Ừ thì về sau lưu sinh và cái lý do.", final_lang="vi")
        session = session_with(decoder)
        committed = finals(speak(session))
        assert committed[0]["transcript"] == "Ừ thì về sau lưu sinh và cái lý do."
        assert session.stats.sentences_from_running_text == 0


class TestLanguageDisagreement:
    def test_the_sentence_is_decoded_again_in_the_language_that_was_spoken(self):
        decoder = ScriptedDecoder(
            partial="その他のタスクの進捗があります", partial_lang="ja",
            final="Các bạn có thể nhận thông tin ở phần bình luận.",
            final_lang="vi",
            retry="その他のタスクの進捗があります。", retry_lang="ja")
        session = session_with(decoder)
        committed = finals(speak(session))
        assert committed[0]["transcript"] == "その他のタスクの進捗があります。"
        assert committed[0]["lang_code"] == "ja"
        assert session.stats.language_retries == 1
        assert session.stats.sentences_from_running_text == 0

    def test_the_retry_is_forced_into_the_running_texts_language(self):
        decoder = ScriptedDecoder(
            partial="はいそうですね", partial_lang="ja",
            final="Vâng đúng rồi.", final_lang="vi",
            retry="はい、そうですね。", retry_lang="ja")
        session_with(decoder)
        speak(session_with(decoder))
        retries = [call for call in decoder.calls
                   if call["beam"] != 1 and call["lang_code"] == "ja"]
        assert retries

    def test_a_retry_that_says_nothing_falls_back_to_the_running_text(self):
        """The only reading that produced anything was in a language nobody
        was speaking, and it has nothing to do with what was on screen."""
        decoder = ScriptedDecoder(
            partial="この件ですけど進捗があります", partial_lang="ja",
            final="Hãy đăng ký kênh để nhận thêm video nhé!", final_lang="vi",
            retry="", retry_lang="ja")
        session = session_with(decoder)
        committed = finals(speak(session))
        assert committed[0]["transcript"] == "この件ですけど進捗があります"
        assert committed[0]["lang_code"] == "ja"
        assert session.stats.language_retries == 1
        assert session.stats.sentences_from_running_text == 1

    def test_a_failed_retry_keeps_a_sentence_that_still_matches(self):
        """A language label can be wrong while the words are right. Only a
        sentence that has also wandered away from the running text is
        replaced by it."""
        decoder = ScriptedDecoder(
            partial="Tranium Access", partial_lang="ja",
            final="Tranium Access", final_lang="vi",
            retry="", retry_lang="ja")
        session = session_with(decoder)
        committed = finals(speak(session))
        assert committed[0]["transcript"] == "Tranium Access"
        assert session.stats.sentences_from_running_text == 0


class TestRendering:
    def test_japanese_never_grows_a_space_on_the_way_through(self):
        """The reference is stitched from several running texts, and the
        splice must not insert what Japanese does not use."""
        decoder = ScriptedDecoder(
            partial="画面共有しましたが", partial_lang="ja",
            final="Chia sẻ màn hình rồi nhé các bạn ơi.", final_lang="vi",
            retry="", retry_lang="ja")
        session = session_with(decoder)
        committed = finals(speak(session))
        assert " " not in committed[0]["transcript"]


class TestHousekeeping:
    def test_the_running_texts_still_reach_the_client(self):
        decoder = ScriptedDecoder(partial="đang nói", partial_lang="vi",
                                  final="Đang nói.", final_lang="vi")
        session = session_with(decoder)
        assert partials(speak(session))

    def test_a_new_meeting_starts_with_nothing_remembered(self):
        decoder = ScriptedDecoder(partial="câu cũ", partial_lang="vi",
                                  final="Câu cũ.", final_lang="vi")
        session = session_with(decoder)
        speak(session)
        session.state = session.state.__class__.AWAITING_HELLO
        session.handle_text(Hello(session_id="second").to_json())
        assert not session.stabilizer.stable(0).has_text

    def test_a_session_without_a_transcriber_does_not_reach_for_one(self):
        vad = ScriptedVAD([0.9])
        session = ServerSession(
            segmenter_factory=lambda: VADSegmenter(vad=vad))
        session.handle_text(Hello(session_id="abc").to_json())
        assert not finals(speak(session))
        assert session.stats.language_retries == 0


# ---------------------------------------------------------------------------
# Splitting an utterance that holds two languages
# ---------------------------------------------------------------------------
class LoudnessProber:
    """A LID that answers by how loud the audio it is handed is.

    The utterance below is quiet for its first half and loud for its second,
    so a stub that reads the amplitude behaves the way a real LID does: its
    answer depends on which span it was given, not on how many times it has
    been called. Counting calls instead made this stub answer differently
    depending on how many running texts had gone past, which is exactly the
    kind of coupling a test should not have.
    """

    def __init__(self, quiet: str = "vi", loud: str = "ja"):
        self.quiet, self.loud = quiet, loud
        self.calls: list = []

    def identify(self, pcm: bytes):
        from server.pipeline.lid import LanguageDecision
        self.calls.append(len(pcm))
        samples = np.frombuffer(pcm, dtype="<i2")
        if samples.size == 0:
            return LanguageDecision("", 0.4, 0.05, "no audio")
        loud = float(np.mean(np.abs(samples.astype(np.float32))))
        return LanguageDecision(self.loud if loud >= 6_000 else self.quiet,
                                0.9, 0.5, "clear")

    def reset(self) -> None:
        self.calls = []


def split_session(decoder, prober):
    vad = ScriptedVAD([0.9])
    session = ServerSession(segmenter_factory=lambda: VADSegmenter(vad=vad),
                            language_identifier=prober,
                            transcriber=Transcriber(decoder=decoder))
    session.handle_text(Hello(session_id="abc").to_json())
    return session


def speak_two_languages(session, quiet: int = 15, loud: int = 15) -> list:
    """One utterance: a quiet turn, then a loud one, no pause between."""
    messages = []
    for _ in range(quiet):
        messages += session.handle_binary(chunk(3_000)).messages
    for _ in range(loud):
        messages += session.handle_binary(chunk(12_000)).messages
    messages += session.finish().messages
    return [json.loads(message) for message in messages]


def utterances(messages) -> list:
    return [message for message in messages if message["type"] == "utterance"]


class TestLanguageSplit:
    def make(self):
        decoder = ScriptedDecoder(partial="đang nói", partial_lang="vi",
                                  final="Một câu.", final_lang="vi")
        return decoder, LoudnessProber()

    def test_an_utterance_holding_two_languages_becomes_two_sentences(self):
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        spoken = utterances(speak_two_languages(session))
        assert session.stats.utterances_split >= 1
        assert len(spoken) > session.buffer.stats.utterances - 1

    def test_the_second_half_is_marked_as_continuing_the_first(self):
        """The translation stage has to know it is reading the middle of a
        turn rather than a new one."""
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        spoken = utterances(speak_two_languages(session))
        assert any(part["continues_previous"] for part in spoken)

    def test_the_halves_do_not_overlap_or_leave_a_gap(self):
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        spoken = utterances(speak_two_languages(session))
        for first, second in zip(spoken, spoken[1:]):
            if second["continues_previous"]:
                assert second["start_ms"] == pytest.approx(first["end_ms"],
                                                           abs=1.0)

    def test_each_half_carries_the_language_found_for_it(self):
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        spoken = utterances(speak_two_languages(session))
        assert {part["lang_code"] for part in spoken} == {"vi", "ja"}

    def test_the_probes_are_counted(self):
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        speak_two_languages(session)
        assert session.stats.language_probes >= 2

    def test_the_running_text_arbitration_is_skipped_on_a_split(self):
        """Each half already carries the language the LID found for it, which
        beats a vote taken across both."""
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        speak_two_languages(session)
        assert session.stats.utterances_split >= 1
        assert session.stats.language_retries == 0

    def test_one_language_throughout_is_left_whole(self):
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        speak(session, speech=20)
        assert session.stats.utterances_split == 0

    def test_without_a_language_model_nothing_is_split(self):
        decoder = ScriptedDecoder(partial="đang nói", partial_lang="vi",
                                  final="Một câu.", final_lang="vi")
        session = session_with(decoder)
        speak_two_languages(session)
        assert session.stats.utterances_split == 0
        assert session.stats.language_probes == 0

    def test_the_environment_can_turn_it_off(self, monkeypatch):
        import server.net.session as session_module
        monkeypatch.setattr(session_module, "LANGUAGE_SPLIT", False)
        decoder, prober = self.make()
        session = split_session(decoder, prober)
        speak_two_languages(session)
        assert session.stats.utterances_split == 0
        assert session.stats.language_probes == 0
