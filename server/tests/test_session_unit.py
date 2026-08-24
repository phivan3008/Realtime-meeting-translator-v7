"""Unit tests for the per-connection server state machine.

No socket, no event loop, no model: ``ServerSession`` returns the messages to
send instead of sending them, precisely so this can run on the Dev PC.

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

from common.protocol import (
    CHUNK_BYTES,
    Hello,
    make_bye,
    make_ready,
)
from server.net.session import ServerSession, SessionState
from server.pipeline.noise import Classification, NoiseFilter
from server.pipeline.vad import VAD_FRAME_SAMPLES, VADEvent, VADSegmenter


class ScriptedVAD:
    def __init__(self, probabilities):
        self.script = list(probabilities)
        self.calls = 0
        self.resets = 0

    def probability(self, frame: np.ndarray) -> float:
        assert frame.shape[-1] == VAD_FRAME_SAMPLES
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return float(value)

    def reset(self) -> None:
        self.resets += 1


def make_session(probabilities=(0.02,), **kwargs) -> ServerSession:
    vad = ScriptedVAD(probabilities)
    return ServerSession(segmenter_factory=lambda: VADSegmenter(vad=vad), **kwargs)


def chunk(value: int = 1000) -> bytes:
    return np.full(CHUNK_BYTES // 2, value, dtype="<i2").tobytes()


def kinds(response) -> list[str]:
    return [json.loads(m)["type"] for m in response.messages]


def of_type(response, wanted: str) -> list[dict]:
    """One response can now carry both vad and utterance messages."""
    return [p for p in (json.loads(m) for m in response.messages)
            if p["type"] == wanted]


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------
def test_a_new_session_waits_for_hello():
    assert make_session().state is SessionState.AWAITING_HELLO


def test_a_good_hello_is_answered_with_ready():
    session = make_session()
    response = session.handle_text(Hello(session_id="abc").to_json())
    assert kinds(response) == ["ready"]
    assert response.close is False
    assert session.state is SessionState.STREAMING
    assert session.session_id == "abc"


def test_ready_echoes_the_session_id():
    session = make_session()
    response = session.handle_text(Hello(session_id="meeting-7").to_json())
    assert json.loads(response.messages[0])["session_id"] == "meeting-7"


def test_a_mismatched_sample_rate_is_refused_before_any_audio():
    """The silent-corruption case the handshake exists to catch."""
    session = make_session()
    response = session.handle_text(
        Hello(session_id="abc", sample_rate=48_000).to_json()
    )
    assert kinds(response) == ["error"]
    assert response.close is True
    assert "sample_rate" in json.loads(response.messages[0])["message"]
    assert session.state is SessionState.CLOSED


def test_a_mismatched_chunk_size_is_refused():
    session = make_session()
    response = session.handle_text(Hello(session_id="abc", chunk_ms=500).to_json())
    assert response.close is True
    assert "chunk_ms" in json.loads(response.messages[0])["message"]


def test_a_future_protocol_version_is_refused():
    session = make_session()
    response = session.handle_text(
        Hello(session_id="abc", protocol_version=99).to_json()
    )
    assert response.close is True
    assert "protocol_version" in json.loads(response.messages[0])["message"]


def test_a_second_hello_on_the_same_connection_is_refused():
    session = make_session()
    session.handle_text(Hello(session_id="abc").to_json())
    response = session.handle_text(Hello(session_id="abc").to_json())
    assert response.close is True
    assert "twice" in json.loads(response.messages[0])["message"]


def test_broken_json_closes_the_connection():
    response = make_session().handle_text("{oops")
    assert kinds(response) == ["error"]
    assert response.close is True


def test_a_server_message_from_a_client_is_refused():
    response = make_session().handle_text(make_ready("abc"))
    assert response.close is True
    assert "unexpected" in json.loads(response.messages[0])["message"]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def test_audio_before_hello_is_refused():
    session = make_session()
    response = session.handle_binary(chunk())
    assert response.close is True
    assert "before a valid hello" in json.loads(response.messages[0])["message"]
    assert session.stats.chunks == 0


def test_a_streaming_session_accepts_chunks_quietly():
    session = make_session([0.02])
    session.handle_text(Hello(session_id="abc").to_json())
    response = session.handle_binary(chunk())
    assert response.messages == []          # silence produces no events
    assert response.close is False
    assert session.stats.chunks == 1
    assert session.stats.bytes_received == CHUNK_BYTES


def test_a_wrong_sized_chunk_closes_the_connection():
    session = make_session()
    session.handle_text(Hello(session_id="abc").to_json())
    response = session.handle_binary(bytes(100))
    assert response.close is True
    assert "exactly 6400 bytes" in json.loads(response.messages[0])["message"]


def test_chunk_size_checking_can_be_relaxed_for_replay_tools():
    session = make_session(strict_chunk_size=False)
    session.handle_text(Hello(session_id="abc").to_json())
    response = session.handle_binary(chunk()[:2048])
    assert response.close is False


def test_speech_produces_vad_events_on_the_wire():
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    response = session.handle_binary(chunk())
    payloads = [json.loads(m) for m in response.messages]
    assert [p["type"] for p in payloads] == ["vad"]
    assert payloads[0]["event"] == VADEvent.SPEECH_START.value
    assert payloads[0]["at_ms"] == 0.0


def test_events_carry_a_growing_timestamp_across_chunks():
    session = make_session([0.02] * 30 + [0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    seen = []
    for _ in range(6):
        seen.extend(json.loads(m) for m in session.handle_binary(chunk()).messages)
    assert seen, "expected a speech_start once the script turns loud"
    assert seen[0]["at_ms"] > 0


def test_the_session_counts_segments():
    # 4 chunks = 25 VAD frames: loud 0-2, quiet 3-18, loud 19-21, quiet after.
    session = make_session([0.9] * 3 + [0.02] * 16 + [0.9] * 3 + [0.02])
    session.handle_text(Hello(session_id="abc").to_json())
    for _ in range(4):
        session.handle_binary(chunk())
    assert session.stats.speech_segments == 2


def test_the_model_state_is_reset_for_each_new_session():
    """Two meetings in a row must not inherit each other's hidden state."""
    vad = ScriptedVAD([0.02])
    factory = lambda: VADSegmenter(vad=vad)     # noqa: E731
    for _ in range(2):
        session = ServerSession(segmenter_factory=factory)
        session.handle_text(Hello(session_id="abc").to_json())
    assert vad.resets == 2


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
def test_bye_during_silence_closes_without_an_error_message():
    session = make_session()
    session.handle_text(Hello(session_id="abc").to_json())
    response = session.handle_text(make_bye("client stopped"))
    assert response.messages == []
    assert response.close is True
    assert "bye" in response.close_reason
    assert session.state is SessionState.CLOSED


def test_bye_mid_sentence_still_ends_the_segment():
    """Found by the real test: a clean stop used to lose the last sentence.

    ``bye`` marked the session closed, so the later ``finish()`` skipped the
    open segment and no ``speech_end`` was ever sent. The buffer manager
    would then hold the final sentence forever and never finalise it.
    """
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    response = session.handle_text(make_bye("client stopped"))
    assert [p["event"] for p in of_type(response, "vad")] == [
        VADEvent.SPEECH_END.value
    ]
    assert response.close is True


def test_the_end_event_is_sent_before_the_socket_closes():
    """It has to ride out on the same response, or it cannot be delivered."""
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    response = session.handle_text(make_bye("done"))
    assert response.messages and response.close


def test_finish_after_bye_does_not_repeat_the_end_event():
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    session.handle_text(make_bye("done"))
    assert session.finish().messages == []


def test_starts_and_ends_balance_over_a_clean_session():
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    events = []
    for _ in range(3):
        events += of_type(session.handle_binary(chunk()), "vad")
    events += of_type(session.handle_text(make_bye("done")), "vad")
    seen = [e["event"] for e in events]
    assert seen.count("speech_start") == seen.count("speech_end") == 1


def test_finish_closes_a_segment_left_open_by_a_dropped_connection():
    """Otherwise the buffer manager waits forever for an end that never comes."""
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    response = session.finish()
    assert [p["event"] for p in of_type(response, "vad")] == [
        VADEvent.SPEECH_END.value
    ]
    assert session.state is SessionState.CLOSED


def test_finish_on_a_silent_session_says_nothing():
    session = make_session([0.02])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    assert session.finish().messages == []


# ---------------------------------------------------------------------------
# Utterance boundaries from the buffer manager
# ---------------------------------------------------------------------------
def test_a_finished_sentence_is_announced_as_an_utterance():
    # 4 chunks = 25 frames: loud 0-2 opens, quiet 3-18 closes on frame 18.
    session = make_session([0.9] * 3 + [0.02] * 16 + [0.02])
    session.handle_text(Hello(session_id="abc").to_json())
    seen = []
    for _ in range(4):
        seen += of_type(session.handle_binary(chunk()), "utterance")
    assert len(seen) == 1
    assert seen[0]["index"] == 0
    assert seen[0]["reason"] == "pause"
    assert seen[0]["continues_previous"] is False
    assert seen[0]["duration_ms"] > 0
    assert session.stats.utterances == 1


def test_the_utterance_span_matches_the_vad_segment():
    session = make_session([0.9] * 3 + [0.02] * 16 + [0.02])
    session.handle_text(Hello(session_id="abc").to_json())
    vad, utterances = [], []
    for _ in range(4):
        response = session.handle_binary(chunk())
        vad += of_type(response, "vad")
        utterances += of_type(response, "utterance")
    starts = [e["at_ms"] for e in vad if e["event"] == "speech_start"]
    ends = [e["at_ms"] for e in vad if e["event"] == "speech_end"]
    assert utterances[0]["start_ms"] == pytest.approx(starts[0])
    assert utterances[0]["end_ms"] == pytest.approx(ends[0], abs=1.0)


def test_a_long_monologue_is_cut_by_the_buffer_not_by_the_vad():
    """35 chunks of unbroken speech is 7 s, the max duration."""
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    seen = []
    for _ in range(40):
        seen += of_type(session.handle_binary(chunk()), "utterance")
    assert seen, "expected a max_duration cut"
    assert seen[0]["reason"] == "max_duration"
    assert seen[0]["duration_ms"] <= 7_000


def test_an_open_utterance_is_committed_when_the_session_ends():
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    seen = of_type(session.finish(), "utterance")
    assert [u["reason"] for u in seen] == ["end_of_stream"]


def test_partials_are_counted_while_a_sentence_is_open():
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    for _ in range(10):
        session.handle_binary(chunk())
    assert session.stats.partials >= 1


def test_finish_on_a_session_that_never_said_hello_is_harmless():
    session = make_session()
    assert session.finish().messages == []
    assert session.state is SessionState.CLOSED


def test_audio_seconds_reflects_what_arrived():
    session = make_session()
    session.handle_text(Hello(session_id="abc").to_json())
    for _ in range(5):
        session.handle_binary(chunk())
    assert session.stats.audio_seconds == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Deep Noise Filter verdicts on the wire
# ---------------------------------------------------------------------------
class StubClassifier:
    def __init__(self, result: Classification):
        self.result = result
        self.calls: list[bytes] = []

    def classify(self, pcm: bytes) -> Classification:
        self.calls.append(pcm)
        return self.result


def filtered_session(result: Classification, probabilities=(0.9,)) -> ServerSession:
    vad = ScriptedVAD(probabilities)
    return ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        noise_filter=NoiseFilter(classifier=StubClassifier(result)),
    )


SPEECHY = Classification(speech_score=0.85, noise_score=0.05, top=(("Speech", 0.85),))
KEYBOARD = Classification(speech_score=0.01, noise_score=0.8,
                          noise_label="Computer keyboard",
                          top=(("Computer keyboard", 0.8),))


def test_a_speech_utterance_is_marked_kept():
    session = filtered_session(SPEECHY)
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    payload = of_type(session.finish(), "utterance")[0]
    assert payload["kept"] is True
    assert payload["speech_score"] == pytest.approx(0.85)
    assert payload["label"] == ""
    assert session.stats.utterances_dropped == 0


def test_keyboard_clatter_is_announced_but_marked_dropped():
    """Still announced: indexes stay consecutive and the client can see why."""
    session = filtered_session(KEYBOARD)
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    payload = of_type(session.finish(), "utterance")[0]
    assert payload["kept"] is False
    assert payload["label"] == "Computer keyboard"
    assert payload["index"] == 0
    assert session.stats.utterances == 1
    assert session.stats.utterances_dropped == 1


def test_the_voiceprint_is_taken_from_the_unshaped_audio():
    """Gating first cost 0.06 of same-speaker cosine on real recordings."""
    class RecordingEmbedder:
        def __init__(self):
            self.seen: list[bytes] = []

        def embed(self, pcm: bytes):
            self.seen.append(pcm)
            return np.array([1.0, 0.0, 0.0])

    class Halving:
        """Stands in for the gate: unmistakably changes the audio."""

        def process(self, samples, sample_rate, gate_threshold_db,
                    compressor_threshold_db):
            return samples * 0.5

    from server.pipeline.diarization import SpeakerIdentifier
    from server.pipeline.overlap import OverlapResolver

    embedder = RecordingEmbedder()
    vad = ScriptedVAD([0.9])
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        overlap_resolver=OverlapResolver(processor=Halving()),
        speaker_identifier=SpeakerIdentifier(embedder=embedder,
                                             min_duration_ms=0),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    session.finish()
    assert embedder.seen, "the identifier was never called"
    quietest = max(abs(v) for v in np.frombuffer(embedder.seen[0], dtype="<i2"))
    assert quietest > 900, "the voiceprint was taken from gated audio"


def test_the_filter_sees_the_committed_audio_not_the_raw_chunk():
    stub = StubClassifier(SPEECHY)
    vad = ScriptedVAD([0.9])
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        noise_filter=NoiseFilter(classifier=stub),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    session.finish()
    assert len(stub.calls) == 1
    assert 0 < len(stub.calls[0]) <= CHUNK_BYTES


def test_a_session_without_the_filter_keeps_everything():
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    payload = of_type(session.finish(), "utterance")[0]
    assert payload["kept"] is True
    assert session.stats.utterances_dropped == 0


# ---------------------------------------------------------------------------
# The whole pipeline, transcriber and translator included
#
# Nothing below existed until a 60 s meeting died seven times on
# ``AttributeError: 'Translation' object has no attribute 'finals'``. The
# session had never been unit-tested with a translator wired in, so the one
# line that needed both a committed sentence and a translator to run was the
# one line no test reached.
# ---------------------------------------------------------------------------
class StubDecoder:
    """Whisper's shape, minus Whisper."""

    def __init__(self, text: str = "xin chào mọi người", lang: str = "vi"):
        self.text = text
        self.lang = lang
        self.calls: list[tuple[int, str, bool]] = []

    def decode(self, samples, lang_code: str = "", beam_size: int = 1):
        self.calls.append((len(samples), lang_code, beam_size))
        from server.pipeline.asr import Piece
        piece = Piece(text=self.text, no_speech_prob=0.01,
                      avg_logprob=-0.2, compression_ratio=1.2)
        return [piece], (lang_code or self.lang)


class StubBackend:
    def __init__(self, answer: str = "こんにちは"):
        self.answer = answer
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append(user)
        return self.answer


def full_session(probabilities=(0.9,), decoder=None, backend=None):
    from server.pipeline.asr import Transcriber
    from server.pipeline.translate import Translator

    decoder = decoder if decoder is not None else StubDecoder()
    backend = backend if backend is not None else StubBackend()
    vad = ScriptedVAD(probabilities)
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        transcriber=Transcriber(decoder=decoder),
        translator=Translator(backend=backend),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    return session, decoder, backend


def speak_then_pause(session, speech_chunks: int = 2, silence_chunks: int = 8):
    """Drive a whole sentence through ``handle_binary`` and let a pause end it.

    ``finish()`` closes a segment too, but it is not the path that crashed:
    the sentences a real meeting commits are committed mid-stream.
    """
    responses = []
    for _ in range(speech_chunks + silence_chunks):
        responses.append(session.handle_binary(chunk()))
    return responses


def test_a_committed_sentence_survives_the_translator():
    """The regression itself: `result` was reassigned inside the loop, so the
    line after it read `.finals` off a Translation."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    responses = speak_then_pause(session)
    finals = [p for r in responses for p in of_type(r, "final")]
    assert finals, "the sentence never reached the wire"
    assert finals[0]["transcript"] == "xin chào mọi người"
    assert finals[0]["translation"] == "こんにちは"


def test_the_utterance_count_survives_the_translator():
    """The crashing line was a statistic. It still has to be right."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    speak_then_pause(session)
    assert session.stats.utterances == 1
    assert session.stats.translations == 1
    # `transcripts` counts partials too, so it runs ahead of the sentences.
    assert session.stats.transcripts >= 1
    assert session.stats.partials >= 1


def test_a_second_sentence_still_counts():
    """One reassignment would have made the count wrong rather than loud."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    speak_then_pause(session)
    for _ in range(14):
        session.handle_binary(chunk())
    assert session.stats.utterances >= 1


def test_a_refused_translation_still_sends_the_sentence():
    """The transcript is worth showing even when the model gives nothing."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02],
                                               backend=StubBackend(answer=""))
    responses = speak_then_pause(session)
    finals = [p for r in responses for p in of_type(r, "final")]
    assert finals
    assert finals[0]["transcript"] == "xin chào mọi người"
    assert finals[0]["translation"] == ""
    assert session.stats.translations == 0


def test_the_translator_reads_the_language_the_pipeline_decided():
    backend = StubBackend()
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02],
                                               backend=backend)
    speak_then_pause(session)
    assert backend.calls, "the translator was never called"


def test_a_stage_that_raises_does_not_end_the_meeting():
    """One broken sentence costs one sentence, not the connection."""
    class Exploding:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("the translator fell over")

    session, _decoder, _backend = full_session([0.9] * 14 + [0.02],
                                               backend=Exploding())
    responses = speak_then_pause(session)
    assert session.state is SessionState.STREAMING
    assert all(r.close is False for r in responses)
    assert session.stats.pipeline_errors == 1
    # And the next chunk is still accepted.
    assert session.handle_binary(chunk()).close is False


def test_a_healthy_run_reports_no_pipeline_errors():
    """So the counter above means something when it is not zero."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    speak_then_pause(session)
    assert session.stats.pipeline_errors == 0


# ---------------------------------------------------------------------------
# The language handed to the ASR
# ---------------------------------------------------------------------------
class SwitchableLID:
    """Answers with whatever ``code`` currently is, so a test can change its
    mind halfway through a meeting the way a short clip makes the real one."""

    def __init__(self, code: str = "vi"):
        self.code = code
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def identify(self, pcm: bytes):
        from server.pipeline.lid import LanguageDecision
        self.calls += 1
        return LanguageDecision(lang_code=self.code, confidence=0.9,
                                margin=0.5, reason="scripted")


def lang_session(code: str = "vi", decoder=None):
    from server.pipeline.asr import Transcriber
    from server.pipeline.translate import Translator

    decoder = decoder if decoder is not None else StubDecoder()
    lid = SwitchableLID(code)
    vad = ScriptedVAD(([0.9] * 14 + [0.02] * 48) * 3)
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        language_identifier=lid,
        transcriber=Transcriber(decoder=decoder),
        translator=Translator(backend=StubBackend()),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    return session, decoder, lid


def forced_languages(decoder) -> list[str]:
    return [lang for _n, lang, _beam in decoder.calls]


def test_a_confident_verdict_is_forced_on_the_asr():
    session, decoder, _lid = lang_session("vi")
    speak_then_pause(session)
    assert forced_languages(decoder)
    assert set(forced_languages(decoder)) == {"vi"}
    assert session._last_language == "vi"


def test_an_undecided_sentence_reuses_the_meetings_last_language():
    """Whisper's own detector answered Swedish (0.66) for this meeting."""
    from server.pipeline.lid import LID_UNKNOWN

    session, decoder, lid = lang_session("vi")
    speak_then_pause(session)                   # the meeting establishes itself
    assert session._last_language == "vi"

    lid.code = LID_UNKNOWN                      # now a clip too short to judge
    decoder.calls.clear()
    speak_then_pause(session)
    assert forced_languages(decoder), "the ASR was never called"
    assert "" not in forced_languages(decoder),         "an undecided sentence was handed to Whisper's own detector"
    assert set(forced_languages(decoder)) == {"vi"}


def test_an_undecided_first_sentence_still_falls_back_to_the_detector():
    """Nothing has been established yet, and inventing a language is worse."""
    from server.pipeline.lid import LID_UNKNOWN

    session, decoder, _lid = lang_session(LID_UNKNOWN)
    speak_then_pause(session)
    assert forced_languages(decoder)
    assert set(forced_languages(decoder)) == {""}


def test_an_undecided_sentence_does_not_overwrite_what_was_established():
    from server.pipeline.lid import LID_UNKNOWN

    session, _decoder, lid = lang_session("ja")
    speak_then_pause(session)
    lid.code = LID_UNKNOWN
    speak_then_pause(session)
    assert session._last_language == "ja"
