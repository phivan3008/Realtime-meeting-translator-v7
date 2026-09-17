"""Unit tests for how the session runs the pipeline stages together.

``test_session_unit.py`` covers the protocol and the basic flow. This covers
what was carried over from the other branches and what was fixed on the way:
a broken stage switched off rather than eating the meeting, the history a
sentence is translated with, an utterance cut on a language change, speaker
labels corrected after the fact, and the running text's language.

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

from common.protocol import CHUNK_BYTES, Hello, make_bye  # noqa: E402
from server.net.session import ServerSession, Stages  # noqa: E402
from server.pipeline.asr import Piece, Transcriber  # noqa: E402
from server.pipeline.language_split import Split  # noqa: E402
from server.pipeline.lid import LanguageDecision  # noqa: E402
from server.pipeline.translate import HISTORY_HEADER, Translator  # noqa: E402
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
        pass


class Decoder:
    """Says the same sentence in whatever language it is asked for."""

    TEXT = {"vi": " xin chào mọi người", "ja": "こんにちは皆さん", "": " xin chào"}

    def __init__(self):
        self.calls: list[dict] = []

    def decode(self, samples, lang_code="", beam_size=1, prompt=None):
        self.calls.append({"lang": lang_code, "beam": beam_size,
                           "seconds": samples.size / 16_000})
        return ([Piece(self.TEXT.get(lang_code, " xin chào"), -0.2, 0.01, 1.3)],
                lang_code or "vi")


class Backend:
    def __init__(self, answer: str = "こんにちは"):
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, messages) -> str:
        self.prompts.append(messages[-1]["content"])
        return self.answer


class LID:
    def __init__(self, code: str = "vi", margin: float = 0.6):
        self.code = code
        self.margin = margin
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def identify(self, pcm: bytes) -> LanguageDecision:
        self.calls += 1
        return LanguageDecision(self.code, 0.9, self.margin, "scripted")


def chunk(value: int = 1000) -> bytes:
    return np.full(CHUNK_BYTES // 2, value, dtype="<i2").tobytes()


def of_type(responses, wanted: str) -> list[dict]:
    if not isinstance(responses, list):
        responses = [responses]
    return [p for r in responses for p in (json.loads(m) for m in r.messages)
            if p["type"] == wanted]


SENTENCE = [0.9] * 40 + [0.02] * 20        # about 1.3 s of speech, then a pause


def session_with(vad_script=SENTENCE * 6, **stages) -> ServerSession:
    vad = ScriptedVAD(vad_script)
    session = ServerSession(segmenter_factory=lambda: VADSegmenter(vad=vad),
                            stages=Stages(**stages), translation_inline=True)
    session.handle_text(Hello(session_id="abc").to_json())
    return session


def speak(session, chunks: int = 12) -> list:
    return [session.handle_binary(chunk()) for _ in range(chunks)]


# ---------------------------------------------------------------------------
# A broken stage
# ---------------------------------------------------------------------------
class Raising:
    def __init__(self, message: str):
        self.message = message
        self.calls = 0

    def reset(self) -> None:
        pass

    def identify(self, pcm: bytes):
        self.calls += 1
        raise RuntimeError(self.message)


def test_a_device_failure_switches_the_stage_off_at_once():
    """On the pod the second call into a stage that failed in cuDNN
    segfaulted the process."""
    broken = Raising("cuDNN Frontend error: No valid execution plans built")
    session = session_with(speaker_identifier=broken,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""))
    responses = speak(session, 24)
    assert broken.calls == 1
    assert session.speaker_identifier is None
    assert session.speaker_history is None
    assert session.stats.stages_disabled == ["speaker"]
    notices = [e for e in of_type(responses, "error") if not e["fatal"]]
    assert len(notices) == 1 and "speaker" in notices[0]["message"]
    assert of_type(responses, "final"), "the sentences stopped with the stage"


def test_an_ordinary_failure_gets_three_chances():
    broken = Raising("index out of range")
    session = session_with(speaker_identifier=broken,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""))
    speak(session, 60)
    assert broken.calls == 3
    assert session.stats.stage_failures == {"speaker": 3}
    assert session.stats.stages_disabled == ["speaker"]


def test_a_broken_language_model_takes_the_splitter_with_it():
    """Both read the same model."""
    broken = Raising("CUDA error: device-side assert triggered")
    session = session_with(language_identifier=broken,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""))
    assert session.language_splitter is not None
    speak(session, 12)
    assert session.language_identifier is None
    assert session.language_splitter is None


def test_a_healthy_meeting_switches_nothing_off():
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=LID())
    speak(session, 24)
    assert session.stats.stage_failures == {}
    assert session.stats.stages_disabled == []


# ---------------------------------------------------------------------------
# The history a sentence is translated with
# ---------------------------------------------------------------------------
def history_part(prompt: str) -> str:
    if HISTORY_HEADER not in prompt:
        return ""
    return prompt.split(HISTORY_HEADER, 1)[1].rsplit("\n\n", 1)[0]


def test_a_sentence_is_never_in_its_own_history():
    """The translation runs later, on another thread. A history read then
    held the sentence being translated - and a model shown the line among
    'the previous lines' hands it back."""
    backend = Backend()
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           translator=Translator(backend=backend))
    speak(session, 30)
    assert len(backend.prompts) >= 2
    assert HISTORY_HEADER not in backend.prompts[0]
    for index, prompt in enumerate(backend.prompts):
        # Sentence n sees the n before it, up to the history's size - each
        # history line opens with who said it.
        lines = [line for line in history_part(prompt).splitlines()
                 if line.startswith("someone")]
        assert len(lines) == min(index, 3), "a line was repeated or missing"


def test_every_sentence_is_in_the_history_once():
    """The session and the translator both remembered each sentence, so a
    three-line history held a line and a half."""
    backend = Backend()
    translator = Translator(backend=backend)
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           translator=translator)
    speak(session, 40)
    assert session._sentences >= 3
    assert len(translator.context.turns) == min(session._sentences,
                                                translator.context.size)
    assert all(turn.translation == "" for turn in translator.context.turns)


def test_the_translator_never_touches_the_shared_history():
    """It runs on its own thread; only the session writes the history."""
    class Recording(Translator):
        def translate(self, *args, **kwargs):
            assert kwargs.get("history") is not None
            return super().translate(*args, **kwargs)

    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           translator=Recording(backend=Backend()))
    speak(session, 24)
    assert session.stats.translations >= 1


# ---------------------------------------------------------------------------
# Two languages in one utterance
# ---------------------------------------------------------------------------
class HalfwaySplitter:
    """Cuts every utterance long enough in the middle."""

    def __init__(self):
        from server.pipeline.language_split import SplitStats
        self.stats = SplitStats()
        self.hangovers: list[float] = []

    def find(self, pcm, hangover_ms=0.0):
        self.hangovers.append(hangover_ms)
        if len(pcm) < 16_000:
            return None
        at = len(pcm) // 2 - (len(pcm) // 2) % 2
        return Split(at=at, first="vi", second="ja", probes=7)

    def reset(self):
        pass


def test_each_half_is_decoded_in_its_own_language():
    decoder = Decoder()
    lid = LID("vi")
    session = session_with(transcriber=Transcriber(decoder=decoder, prompt=""),
                           language_identifier=lid)
    session.language_splitter = HalfwaySplitter()
    responses = speak(session, 12)
    finals = of_type(responses, "final")
    assert [f["lang_code"] for f in finals[:2]] == ["vi", "ja"]
    assert finals[1]["transcript"] == "こんにちは皆さん"
    assert session.stats.language_splits >= 1


def test_the_halves_are_not_asked_about_their_language_again():
    """The review probes already answered for each half."""
    lid = LID("vi")
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    session.language_splitter = HalfwaySplitter()
    calls_before = []

    original = session._analyse

    def spy(utterance, spent, asr_id=None, language=""):
        calls_before.append(lid.calls)
        found = original(utterance, spent, asr_id=asr_id, language=language)
        assert lid.calls == calls_before[-1], "the LID was asked again"
        return found

    session._analyse = spy
    speak(session, 12)
    assert calls_before


def test_the_splitter_is_told_how_much_of_the_end_is_silence():
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=LID())
    splitter = HalfwaySplitter()
    session.language_splitter = splitter
    speak(session, 12)
    assert splitter.hangovers and splitter.hangovers[0] > 0


def test_a_split_leaves_no_streaming_state_behind():
    transcriber = Transcriber(decoder=Decoder(), prompt="")
    session = session_with(transcriber=transcriber, language_identifier=LID())
    session.language_splitter = HalfwaySplitter()
    speak(session, 12)
    assert transcriber._states == {} or all(
        key == session._open_asr_id for key in transcriber._states)


# ---------------------------------------------------------------------------
# Speaker labels, corrected later
# ---------------------------------------------------------------------------
class Voices:
    """Hands out one fixed voiceprint per call, in a scripted order."""

    def __init__(self, *labels):
        from server.pipeline.diarization import Assignment
        self.labels = list(labels)
        self.calls = 0
        self.Assignment = Assignment

    def reset(self) -> None:
        pass

    def identify(self, pcm: bytes):
        label = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        vector = np.zeros(8)
        vector[0] = 1.0
        return self.Assignment(label, 0.9, False, "scripted",
                               embedding=vector)


def test_corrected_labels_are_sent_as_a_speakers_message():
    session = session_with(
        transcriber=Transcriber(decoder=Decoder(), prompt=""),
        speaker_identifier=Voices("Speaker_01", "Speaker_02", "Speaker_01"))
    from server.pipeline.reclustering import SpeakerHistory
    session.speaker_history = SpeakerHistory(every=3, confirmations=1)
    session.send_speaker_corrections = True
    responses = speak(session, 30)
    corrections = of_type(responses, "speakers")
    assert corrections, "no correction was sent"
    assert corrections[0]["labels"] == {"2": "Speaker_01"}
    assert session.stats.speaker_corrections == 1


# ---------------------------------------------------------------------------
# Running text
# ---------------------------------------------------------------------------
class ScriptedLID:
    """Answers each call from a script, then repeats the last answer."""

    def __init__(self, *codes: str):
        self.codes = list(codes)
        self.calls = 0

    def reset(self) -> None:
        pass

    def identify(self, pcm: bytes) -> LanguageDecision:
        code = self.codes[min(self.calls, len(self.codes) - 1)]
        self.calls += 1
        return LanguageDecision(code, 0.9, 0.6 if code else 0.0, "scripted")


def running_languages(session, lid, chunks):
    decoder = session.transcriber.decoder
    speak(session, chunks)
    return [call["lang"] for call in decoder.calls if call["beam"] == 1]


def test_one_window_does_not_fix_the_running_texts_language():
    """Fixed on the first short window, about thirty sentences of a real
    meeting ran as Vietnamese inventions over Japanese speech."""
    lid = ScriptedLID("vi", "ja", "ja", "ja", "ja")
    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    languages = running_languages(session, lid, 15)
    assert languages[:2] == ["vi", "ja"]
    assert set(languages[2:]) == {"ja"}
    assert session._open_language == "ja"


def test_two_confident_windows_change_a_fixed_language():
    lid = ScriptedLID("vi", "vi", "vi", "ja", "ja", "ja")
    session = session_with(vad_script=[0.9] * 300,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    languages = running_languages(session, lid, 20)
    assert languages[:4] == ["vi", "vi", "vi", "vi"]
    assert set(languages[4:]) == {"ja"}
    assert session.stats.running_language_changes == 1
    assert session.transcriber.stats.language_resets == 1


def test_one_stray_window_does_not_change_a_fixed_language():
    lid = ScriptedLID("vi", "vi", "ja", "vi", "vi")
    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    languages = running_languages(session, lid, 15)
    assert set(languages) == {"vi"}
    assert session.stats.running_language_changes == 0


def test_the_vote_starts_again_with_each_utterance():
    lid = ScriptedLID("vi")
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    speak(session, 8)                  # two running-text windows
    assert session._open_language == "vi"
    speak(session, 1)                  # the pause ends the sentence
    assert session._open_language == ""
    assert session._language_run == ("", 0)


def test_a_sure_lid_on_the_whole_sentence_wins_over_the_running_text():
    """The sentence is decoded whole again in the LID's language."""
    # Two running-text windows say Vietnamese; the whole sentence is Japanese.
    lid = ScriptedLID("vi", "vi", "ja")
    decoder = Decoder()
    session = session_with(transcriber=Transcriber(decoder=decoder, prompt=""),
                           language_identifier=lid)
    session.language_splitter = None       # it would ask the LID as well
    responses = speak(session, 10)
    finals = of_type(responses, "final")
    assert finals and finals[0]["lang_code"] == "ja"
    assert finals[0]["transcript"] == "こんにちは皆さん"
    assert session.stats.language_flips == 1


def test_an_unsure_lid_leaves_the_running_texts_language():
    class UnsureOnWholeSentences(ScriptedLID):
        def identify(self, pcm: bytes) -> LanguageDecision:
            if len(pcm) > 64_000:        # over two seconds: the whole sentence
                self.calls += 1
                return LanguageDecision("", 0.5, 0.05, "too close")
            return super().identify(pcm)

    lid = UnsureOnWholeSentences("ja")
    session = session_with(vad_script=[0.9] * 90 + [0.02] * 30,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    session.language_splitter = None
    session._last_language = "vi"
    responses = speak(session, 20)
    finals = of_type(responses, "final")
    assert finals and finals[0]["lang_code"] == "ja"
    assert session.stats.language_flips == 0


def test_an_unsure_lid_is_asked_again_on_the_next_window():
    lid = LID("", margin=0.0)
    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    speak(session, 6)
    assert lid.calls == session.stats.partials


def test_japanese_running_text_has_no_spaces():
    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=LID("ja"))
    responses = speak(session, 8)
    partials = of_type(responses, "partial")
    assert partials
    assert all(" " not in p["transcript"] for p in partials)


def test_sentences_and_running_texts_are_counted_apart():
    session = session_with(transcriber=Transcriber(decoder=Decoder(), prompt=""))
    responses = speak(session, 12)
    assert session.stats.transcripts == len(of_type(responses, "final"))
    assert session.stats.running_texts >= 1


def test_an_emptied_running_text_is_cleared_once():
    class GoesQuiet(Decoder):
        def decode(self, samples, lang_code="", beam_size=1, prompt=None):
            self.calls.append({})
            if len(self.calls) == 1:
                return [Piece(" xin", -0.2, 0.01, 1.3)], "vi"
            return [], "vi"

    session = session_with(vad_script=[0.9] * 300,
                           transcriber=Transcriber(decoder=GoesQuiet(),
                                                   prompt=""))
    partials = of_type(speak(session, 10), "partial")
    assert [p["transcript"] for p in partials] == ["xin", ""]


def test_the_final_is_told_where_the_speech_ended():
    class Spy(Transcriber):
        def __init__(self):
            super().__init__(decoder=Decoder(), prompt="")
            self.ends = []

        def finish_utterance(self, full_pcm, **kwargs):
            self.ends.append((len(full_pcm) / 32_000,
                              kwargs["speech_end_seconds"]))
            return super().finish_utterance(full_pcm, **kwargs)

    spy = Spy()
    session = session_with(transcriber=spy)
    speak(session, 12)
    assert spy.ends
    duration, speech_end = spy.ends[0]
    # The VAD forwards its 480 ms of hangover; the final is told about it.
    assert duration - speech_end == pytest.approx(0.48, abs=0.01)


def test_the_goodbye_carries_the_hangover_it_already_forwarded():
    class Spy(Transcriber):
        def __init__(self):
            super().__init__(decoder=Decoder(), prompt="")
            self.ends = []

        def finish_utterance(self, full_pcm, **kwargs):
            self.ends.append((len(full_pcm) / 32_000,
                              kwargs["speech_end_seconds"]))
            return super().finish_utterance(full_pcm, **kwargs)

    spy = Spy()
    session = session_with(vad_script=[0.9] * 20 + [0.02] * 100,
                           transcriber=spy)
    # A chunk is 6.25 VAD frames: five of them are 20 loud frames and then
    # 11 quiet ones, short of the 16 that would close the segment.
    speak(session, 5)
    session.handle_text(make_bye("done"))
    duration, speech_end = spy.ends[0]
    assert duration - speech_end == pytest.approx(11 * 0.032, abs=0.01)


# ---------------------------------------------------------------------------
# Speaker corrections are measured, not sent, by default
# ---------------------------------------------------------------------------
def test_corrections_are_not_sent_by_default(caplog):
    """On a real meeting of four people they made the labels worse."""
    import logging
    from server.pipeline.reclustering import SpeakerHistory

    session = session_with(
        transcriber=Transcriber(decoder=Decoder(), prompt=""),
        speaker_identifier=Voices("Speaker_01", "Speaker_02", "Speaker_01"))
    assert session.send_speaker_corrections is False
    session.speaker_history = SpeakerHistory(every=3, confirmations=1)
    with caplog.at_level(logging.INFO):
        responses = speak(session, 30)
    assert of_type(responses, "speakers") == []
    assert session.stats.speaker_corrections == 0
    stats = session.speaker_history.stats
    assert stats.runs >= 1
    assert stats.would_move == 1
    assert "not sent" in caplog.text
    # And nothing on the history itself was relabelled.
    assert [v.label for v in session.speaker_history.voices][:2] == [
        "Speaker_01", "Speaker_02"]


# ---------------------------------------------------------------------------
# The running text's language changes: cut there
#
# 09-17, sentence #6: "Đi kiểm chứng tiếp" in the running text for three
# seconds, then "ステップ011の方は" - and the sentence held only the Japanese.
# The cut at the end of the sentence had declined.
# ---------------------------------------------------------------------------
class MidSplitter:
    """Finds the boundary at a fixed point, and records what it was asked."""

    def __init__(self, first="vi", second="ja", at_ms=600.0):
        from server.pipeline.language_split import SplitStats
        self.stats = SplitStats()
        self.first, self.second, self.at_ms = first, second, at_ms
        self.asked: list = []

    def find(self, pcm, hangover_ms=0.0):
        self.asked.append((len(pcm), hangover_ms))
        at = int(self.at_ms * 32)
        if len(pcm) <= at:
            return None
        return Split(at=at, first=self.first, second=self.second, probes=7)

    def reset(self):
        pass


def changing_session(splitter, script=("vi", "vi", "vi", "ja", "ja", "ja")):
    lid = ScriptedLID(*script)
    session = session_with(vad_script=[0.9] * 300,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    session.language_splitter = splitter
    return session, lid


def test_a_running_text_that_changes_language_is_cut_there():
    splitter = MidSplitter()
    session, _lid = changing_session(splitter)
    responses = speak(session, 20)
    finals = of_type(responses, "final")
    assert finals, "the first turn was never committed"
    assert finals[0]["lang_code"] == "vi"
    assert finals[0]["transcript"] == "xin chào mọi người"
    utterances = of_type(responses, "utterance")
    assert utterances[0]["reason"] == "language_change"
    assert utterances[0]["duration_ms"] == pytest.approx(600, abs=1)
    assert session.stats.running_language_cuts == 1
    # Asked about the open audio, which has no hangover in it yet.
    assert splitter.asked[0][1] == 0.0


def test_the_turn_after_the_cut_goes_on_in_its_own_language():
    session, _lid = changing_session(MidSplitter())
    speak(session, 30)
    assert session._open_language == "ja"
    decoder = session.transcriber.decoder
    after_cut = [call["lang"] for call in decoder.calls
                 if call["beam"] == 1][-2:]
    assert set(after_cut) == {"ja"}
    assert session.buffer.open_index == 1


def test_no_cut_when_the_splitter_finds_nothing():
    class Declines(MidSplitter):
        def find(self, pcm, hangover_ms=0.0):
            self.asked.append((len(pcm), hangover_ms))
            return None

    splitter = Declines()
    session, _lid = changing_session(splitter)
    responses = speak(session, 20)
    assert splitter.asked, "it was never asked"
    assert of_type(responses, "final") == []
    assert session.stats.running_language_cuts == 0
    assert session.stats.running_language_changes == 1


def test_no_cut_when_the_splitter_hears_other_languages():
    """The cut has to agree with what the running text heard."""
    session, _lid = changing_session(MidSplitter(first="ja", second="vi"))
    responses = speak(session, 20)
    assert of_type(responses, "final") == []
    assert session.stats.running_language_cuts == 0


def test_the_first_fix_of_a_language_is_not_a_change():
    splitter = MidSplitter()
    session, _lid = changing_session(splitter, script=("ja",))
    speak(session, 20)
    assert splitter.asked == []
    assert session.stats.running_language_cuts == 0


# ---------------------------------------------------------------------------
# How sure a language has to be to overrule the running text
# ---------------------------------------------------------------------------
class MarginLID(ScriptedLID):
    """Scripted languages, with one margin for the whole-sentence answer."""

    def __init__(self, *codes, sentence_code="ja", sentence_margin=0.4):
        super().__init__(*codes)
        self.sentence_code = sentence_code
        self.sentence_margin = sentence_margin

    def identify(self, pcm: bytes) -> LanguageDecision:
        if len(pcm) > 130_000:           # longer than any 4 s window
            self.calls += 1
            return LanguageDecision(self.sentence_code, 0.7,
                                    self.sentence_margin, "whole")
        return super().identify(pcm)


def test_a_weak_whole_sentence_answer_leaves_the_running_text():
    """09-17 replay, sentence #2: six seconds of correct Vietnamese running
    text, decoded again as Japanese on a 0.30-margin answer."""
    lid = MarginLID("vi", sentence_code="ja", sentence_margin=0.35)
    session = session_with(vad_script=[0.9] * 150 + [0.02] * 30,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    session.language_splitter = None
    finals = of_type(speak(session, 32), "final")
    assert finals and finals[0]["lang_code"] == "vi"
    assert session.stats.language_flips == 0


def test_a_strong_whole_sentence_answer_still_wins():
    lid = MarginLID("vi", sentence_code="ja", sentence_margin=0.8)
    session = session_with(vad_script=[0.9] * 150 + [0.02] * 30,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    session.language_splitter = None
    finals = of_type(speak(session, 32), "final")
    assert finals and finals[0]["lang_code"] == "ja"
    assert session.stats.language_flips == 1


def test_an_unsure_window_is_not_forced_before_the_language_is_fixed():
    """09-17 #28: 'Bên mặt của nó sẽ là' over Japanese speech, from single
    windows the LID was not sure of."""
    class Weak(ScriptedLID):
        def identify(self, pcm: bytes) -> LanguageDecision:
            self.calls += 1
            return LanguageDecision("vi", 0.6, 0.35, "weak")

    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=Weak())
    session.language_splitter = None
    session._last_language = "ja"
    speak(session, 4)                       # one window: not fixed yet
    decoder = session.transcriber.decoder
    assert [c["lang"] for c in decoder.calls if c["beam"] == 1] == ["ja"]


def test_a_sure_window_is_used_before_the_language_is_fixed():
    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=ScriptedLID("ja"))
    session.language_splitter = None
    session._last_language = "vi"
    speak(session, 4)
    decoder = session.transcriber.decoder
    assert [c["lang"] for c in decoder.calls if c["beam"] == 1] == ["ja"]
