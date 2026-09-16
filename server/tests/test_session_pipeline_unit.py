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
    responses = speak(session, 30)
    corrections = of_type(responses, "speakers")
    assert corrections, "no correction was sent"
    assert corrections[0]["labels"] == {"2": "Speaker_01"}
    assert session.stats.speaker_corrections == 1


# ---------------------------------------------------------------------------
# Running text
# ---------------------------------------------------------------------------
def test_the_running_texts_language_is_decided_once_per_utterance():
    """Decodes forced into different languages never agree, so nothing
    would ever be committed."""
    lid = LID("vi")
    session = session_with(vad_script=[0.9] * 200,
                           transcriber=Transcriber(decoder=Decoder(), prompt=""),
                           language_identifier=lid)
    speak(session, 15)
    assert session.stats.partials >= 3
    assert lid.calls == 1


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
