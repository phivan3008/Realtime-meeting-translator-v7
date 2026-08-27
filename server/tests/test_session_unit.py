"""Unit tests for the per-connection server state machine.

No socket, no event loop, no model: ``ServerSession`` returns the messages to
send instead of sending them, precisely so this can run on the Dev PC.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import json
import logging
import sys
import time
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
from server.pipeline.diarization import SpeakerIdentifier, SpeakerRegistry
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
        # Inline: a worker thread would make these tests depend on
        # scheduling, and the queue's own tests cover the threaded path.
        translation_inline=True,
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


def test_the_translation_follows_the_sentence_it_belongs_to():
    """Two messages now: the sentence goes out at once and the translation
    catches up, matched by sentence_id."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    responses = speak_then_pause(session)
    finals = [p for r in responses for p in of_type(r, "final")]
    translations = [p for r in responses for p in of_type(r, "translation")]
    assert translations, "the translation never arrived"
    assert translations[0]["sentence_id"] == finals[0]["sentence_id"]
    assert translations[0]["translation"] == "こんにちは"


def test_the_sentence_does_not_wait_for_the_translation():
    """The whole point: an LLM call must not sit on the audio path."""
    class Slow:
        def complete(self, system: str, user: str) -> str:
            raise AssertionError("the translator was called on the audio path")

    from server.pipeline.asr import Transcriber
    from server.pipeline.translate import Translator

    vad = ScriptedVAD([0.9] * 14 + [0.02])
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        transcriber=Transcriber(decoder=StubDecoder()),
        translator=Translator(backend=Slow()),
        translation_inline=False,      # queued, never run: no thread started
    )
    session.handle_text(Hello(session_id="abc").to_json())
    responses = speak_then_pause(session)
    finals = [p for r in responses for p in of_type(r, "final")]
    assert finals, "the sentence waited for a translation that never came"
    assert finals[0]["transcript"] == "xin chào mọi người"


def test_sentence_ids_do_not_repeat_across_segments():
    """Utterance indexes restart with each speech segment, so they cannot be
    used to match a translation to its sentence."""
    session, _decoder, _backend = full_session()
    responses = speak_then_pause(session)
    for _ in range(3):
        responses += speak_then_pause(session)
    ids = [p["sentence_id"] for r in responses for p in of_type(r, "final")]
    assert len(ids) == len(set(ids)), ids
    assert ids == sorted(ids)


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
    translations = [p for r in responses for p in of_type(r, "translation")]
    assert finals
    assert finals[0]["transcript"] == "xin chào mọi người"
    assert translations, "a refusal has to say so, not simply never arrive"
    assert translations[0]["translation"] == ""
    assert translations[0]["reason"]
    assert session.stats.translations == 0
    assert session.stats.translations_dropped == 1


def test_the_translator_reads_the_language_the_pipeline_decided():
    backend = StubBackend()
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02],
                                               backend=backend)
    speak_then_pause(session)
    assert backend.calls, "the translator was never called"


def test_a_stage_that_raises_costs_that_stage_and_nothing_else():
    """It used to cost the sentence. On the pod that meant every sentence,
    because the failure was in the stage rather than in the audio."""
    class Exploding:
        def judge(self, pcm: bytes):
            raise RuntimeError("the noise filter fell over")

    vad = ScriptedVAD([0.9] * 14 + [0.02])
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        noise_filter=Exploding(),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    responses = speak_then_pause(session)
    assert session.state is SessionState.STREAMING
    assert all(r.close is False for r in responses)
    assert session.stats.stage_failures == {"noise": 1}
    assert session.stats.pipeline_errors == 0, "the sentence was thrown away"
    # The utterance still went out, without the filter's verdict on it.
    assert of_type(responses[-1], "utterance") or session.stats.utterances == 1
    # And the next chunk is still accepted.
    assert session.handle_binary(chunk()).close is False


def test_a_translator_that_raises_costs_one_translation(caplog):
    """It runs off the audio path now, so it cannot reach the pipeline error
    handler at all - and must not take the worker down with it either."""
    class Exploding:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("the translator fell over")

    session, _decoder, _backend = full_session([0.9] * 14 + [0.02],
                                               backend=Exploding())
    with caplog.at_level(logging.ERROR):
        responses = speak_then_pause(session)
    assert session.state is SessionState.STREAMING
    assert session.stats.pipeline_errors == 0
    finals = [p for r in responses for p in of_type(r, "final")]
    translations = [p for r in responses for p in of_type(r, "translation")]
    assert finals, "the sentence should still have gone out"
    assert translations[0]["reason"] == "the translator raised"


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


# ---------------------------------------------------------------------------
# Where the time goes
#
# Every stage runs on the thread that reads the socket, so a slow sentence is
# a stalled connection. One run showed VAD events arriving 12.6 s late and
# nothing on the server said which stage had taken it.
# ---------------------------------------------------------------------------
class SlowDecoder(StubDecoder):
    def __init__(self, seconds: float, **kwargs):
        super().__init__(**kwargs)
        self.seconds = seconds

    def decode(self, samples, lang_code: str = "", beam_size: int = 1):
        time.sleep(self.seconds)
        return super().decode(samples, lang_code, beam_size)


def test_each_stage_is_charged_for_its_own_time():
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    speak_then_pause(session)
    # No "translate": it runs off this thread now, which is the point.
    assert set(session.stats.stage_seconds) >= {"asr"}
    assert "translate" not in session.stats.stage_seconds
    assert all(value >= 0 for value in session.stats.stage_seconds.values())


def test_the_slowest_sentence_is_remembered():
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    speak_then_pause(session)
    assert session.stats.slowest_utterance_seconds > 0


def test_a_slow_sentence_names_the_stage_that_took_the_time(caplog):
    """The log has to say *which* stage, or it only repeats what the client
    already knew: that the connection stalled."""
    from server.pipeline.asr import Transcriber
    from server.pipeline.translate import Translator

    vad = ScriptedVAD([0.9] * 14 + [0.02])
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        transcriber=Transcriber(decoder=SlowDecoder(1.1)),
        translator=Translator(backend=StubBackend()),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    with caplog.at_level(logging.WARNING, logger="server.net.session"):
        speak_then_pause(session)
    assert "held the socket for" in caplog.text
    assert "asr took" in caplog.text


def test_a_fast_sentence_says_nothing(caplog):
    """A warning on every sentence is a warning on none."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    with caplog.at_level(logging.WARNING, logger="server.net.session"):
        speak_then_pause(session)
    assert "held the socket" not in caplog.text


def test_the_threshold_is_a_few_chunks_of_audio():
    """At 200 ms a chunk, a second is five chunks the socket did not read."""
    from server.net.session import SLOW_UTTERANCE_SECONDS
    assert SLOW_UTTERANCE_SECONDS == 1.0


# ---------------------------------------------------------------------------
# The last sentence of the meeting
#
# It is committed when the client says goodbye, queued for translation there,
# and the socket closes as soon as that is answered. On the tenth end-to-end
# run the sentence arrived and its translation never did.
# ---------------------------------------------------------------------------
def test_the_last_sentence_gets_its_translation_before_the_socket_closes():
    session, _decoder, _backend = full_session([0.9] * 14)
    for _ in range(3):
        session.handle_binary(chunk())
    response = session.handle_text(make_bye("client stopped"))
    finals = of_type(response, "final")
    translations = of_type(response, "translation")
    assert finals, "the last sentence never made it"
    assert response.close is True
    assert [t["sentence_id"] for t in translations] == \
        [f["sentence_id"] for f in finals]


def test_every_sentence_committed_on_bye_is_answered():
    """Answered, not necessarily translated - but never simply absent."""
    session, _decoder, _backend = full_session([0.9] * 14,
                                               backend=StubBackend(answer=""))
    for _ in range(3):
        session.handle_binary(chunk())
    response = session.handle_text(make_bye(""))
    finals = of_type(response, "final")
    translations = of_type(response, "translation")
    assert len(translations) == len(finals)
    assert all(t["reason"] for t in translations)


def test_finishing_after_a_bye_repeats_nothing():
    """finish() still runs in the server's finally block."""
    session, _decoder, _backend = full_session([0.9] * 14)
    for _ in range(3):
        session.handle_binary(chunk())
    session.handle_text(make_bye(""))
    assert session.finish().messages == []


def test_a_dropped_connection_still_answers_its_last_sentence():
    """No bye at all: the finally block has to do the same job."""
    session, _decoder, _backend = full_session([0.9] * 14)
    for _ in range(3):
        session.handle_binary(chunk())
    response = session.finish()
    finals = of_type(response, "final")
    translations = of_type(response, "translation")
    assert finals
    assert [t["sentence_id"] for t in translations] == \
        [f["sentence_id"] for f in finals]


# ---------------------------------------------------------------------------
# The running text is where the time actually goes
#
# It runs every 600 ms on the whole open utterance, so a seven-second sentence
# is decoded eleven times at growing lengths. Measuring only the finals, a
# ten-minute run reported "slowest sentence 0.4 s" while the connection was
# stalled for eleven seconds inside the partial path.
# ---------------------------------------------------------------------------
def test_the_running_text_is_timed_too():
    session, _decoder, _backend = full_session([0.9] * 30)
    for _ in range(6):
        session.handle_binary(chunk())
    assert "partial_asr" in session.stats.stage_seconds
    assert session.stats.slowest_partial_seconds > 0


def test_the_running_text_is_counted_apart_from_the_sentences():
    """Folded together, a slow partial would read as a slow sentence and the
    fix would be aimed at the wrong path."""
    session, _decoder, _backend = full_session([0.9] * 14 + [0.02])
    speak_then_pause(session)
    assert "asr" in session.stats.stage_seconds
    assert "partial_asr" in session.stats.stage_seconds
    assert session.stats.stage_seconds["asr"] != \
        session.stats.stage_seconds["partial_asr"]


def test_slow_running_text_names_itself(caplog):
    from server.pipeline.asr import Transcriber
    from server.pipeline.translate import Translator

    vad = ScriptedVAD([0.9] * 30)
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        transcriber=Transcriber(decoder=SlowDecoder(1.1)),
        translator=Translator(backend=StubBackend()),
        translation_inline=True,
    )
    session.handle_text(Hello(session_id="abc").to_json())
    with caplog.at_level(logging.WARNING, logger="server.net.session"):
        for _ in range(6):
            session.handle_binary(chunk())
    assert "running text" in caplog.text
    assert "held the socket for" in caplog.text
    assert "partial_asr took" in caplog.text


def test_fast_running_text_says_nothing(caplog):
    session, _decoder, _backend = full_session([0.9] * 30)
    with caplog.at_level(logging.WARNING, logger="server.net.session"):
        for _ in range(6):
            session.handle_binary(chunk())
    assert "running text" not in caplog.text


def test_the_partial_language_lookup_is_timed_as_well():
    """It runs as often as the partial decode does."""
    from server.pipeline.asr import Transcriber

    vad = ScriptedVAD([0.9] * 30)
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        language_identifier=SwitchableLID("vi"),
        transcriber=Transcriber(decoder=StubDecoder()),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    for _ in range(6):
        session.handle_binary(chunk())
    assert "partial_language" in session.stats.stage_seconds


# ---------------------------------------------------------------------------
# The cap has to reach the decoder
# ---------------------------------------------------------------------------
def test_the_partial_decoder_never_sees_more_than_the_cap():
    """Capping the window in the buffer is no use if the session hands the
    full one to Whisper anyway. 97.8 s of a ten-minute run went here."""
    from server.config import PARTIAL_WINDOW_SECONDS, SAMPLE_RATE
    from server.pipeline.asr import Transcriber

    vad = ScriptedVAD([0.9] * 200)
    decoder = StubDecoder()
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        transcriber=Transcriber(decoder=decoder),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    for _ in range(40):                      # 8 s of audio, well past the cap
        session.handle_binary(chunk())

    # Partials only. A max-duration cut commits a sentence in the middle of
    # this, and that one is decoded in full on purpose - an earlier version of
    # this test measured both together and failed on the sentence.
    from server.config import ASR_BEAM_SIZE_PARTIAL
    partials = [samples for samples, _lang, beam in decoder.calls
                if beam == ASR_BEAM_SIZE_PARTIAL]
    assert partials, "the running text never ran"
    longest = max(partials)
    assert longest <= PARTIAL_WINDOW_SECONDS * SAMPLE_RATE + 1, \
        f"{longest / SAMPLE_RATE:.1f} s reached the decoder"


def test_the_committed_sentence_is_still_decoded_in_full():
    """The cap is for the running text. A sentence cut short would be a
    sentence with words missing from it for good."""
    from server.config import SAMPLE_RATE
    from server.pipeline.asr import Transcriber

    vad = ScriptedVAD([0.9] * 200 + [0.02] * 200)
    decoder = StubDecoder()
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        transcriber=Transcriber(decoder=decoder),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    for _ in range(40):
        session.handle_binary(chunk())
    for _ in range(20):
        session.handle_binary(chunk())

    finals = [samples for samples, _lang, beam in decoder.calls if beam > 1]
    assert finals, "no sentence was committed"
    assert max(finals) / SAMPLE_RATE > 5.0, \
        "the committed sentence was cut down to the partial window"


# ---------------------------------------------------------------------------
# A second voice ends the sentence
# ---------------------------------------------------------------------------
class VoiceByVolume:
    """A voiceprint that is just loud-or-quiet, so two "people" fit in a test."""

    def embed(self, pcm: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        loud = float(np.abs(samples).mean()) > 4_000
        return np.array([1.0, 0.0]) if loud else np.array([0.0, 1.0])


@pytest.fixture
def cutting_on(monkeypatch):
    """The boundary ships off - one-second voiceprints did not separate
    speakers on a real meeting. These test the mechanism, not the default."""
    monkeypatch.setattr("server.net.session.SPEAKER_CHANGE_ENABLED", True)


def two_voice_session(**kwargs) -> ServerSession:
    vad = ScriptedVAD((0.9,))
    return ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        speaker_identifier=SpeakerIdentifier(embedder=VoiceByVolume()),
        **kwargs,
    )


def speak(session, chunks: int, level: int) -> list[dict]:
    """Feed uninterrupted speech at one volume; return every message sent."""
    payloads = []
    for _ in range(chunks):
        payloads += [json.loads(m)
                     for m in session.handle_binary(chunk(level)).messages]
    return payloads


LOUD, QUIET = 8_000, 500


def test_a_second_voice_ends_the_sentence_without_a_pause(cutting_on):
    """The VAD needs 500 ms of silence; people do not leave that much."""
    session = two_voice_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 12, LOUD) + speak(session, 12, QUIET)
    reasons = [p["reason"] for p in payloads if p["type"] == "utterance"]
    assert "speaker_change" in reasons
    assert session.stats.speaker_changes == 1


def test_one_voice_talking_on_is_not_cut(cutting_on):
    session = two_voice_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 24, LOUD)
    reasons = [p["reason"] for p in payloads if p["type"] == "utterance"]
    assert "speaker_change" not in reasons
    assert session.stats.speaker_changes == 0


def test_the_two_voices_become_two_sentences(cutting_on):
    """The point of the cut: each half gets its own ASR and language pass."""
    session = two_voice_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = (speak(session, 12, LOUD) + speak(session, 12, QUIET)
                + [json.loads(m) for m in session.finish().messages])
    utterances = [p for p in payloads if p["type"] == "utterance"]
    assert len(utterances) >= 2, "both voices came out as one sentence"


def test_the_cut_is_off_unless_asked_for():
    """Measured on a real meeting, one-second voiceprints put 53% of all
    comparisons over the threshold, which shredded the transcript."""
    session = two_voice_session()
    assert session.speaker_change is None
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 12, LOUD) + speak(session, 12, QUIET)
    reasons = [p["reason"] for p in payloads if p["type"] == "utterance"]
    assert "speaker_change" not in reasons


def test_without_a_speaker_identifier_there_is_nothing_to_compare(cutting_on):
    assert make_session().speaker_change is None


# ---------------------------------------------------------------------------
# Second thoughts about who said what
# ---------------------------------------------------------------------------
def merged_session() -> ServerSession:
    """A session whose live matcher merges everybody, as reported from a real
    meeting: after four minutes every sentence was Speaker_01.

    The buffer cuts every second so a few chunks make several sentences -
    with nobody pausing, the only other boundary is the seven-second limit.
    """
    from server.pipeline.asr import Transcriber
    from server.pipeline.buffer import BufferManager

    vad = ScriptedVAD((0.9,))
    return ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        buffer_factory=lambda: BufferManager(max_duration_ms=1_000,
                                             partial_interval_ms=600,
                                             split_search_ms=200),
        transcriber=Transcriber(decoder=StubDecoder()),
        speaker_identifier=SpeakerIdentifier(
            embedder=VoiceByVolume(),
            registry=SpeakerRegistry(match_threshold=-1.0)),
    )


def test_the_meeting_is_clustered_again_and_the_labels_come_back():
    """The live matcher merged two voices; clustering pulls them apart."""
    session = merged_session()
    session.speaker_history.every = 1
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 10, LOUD) + speak(session, 10, QUIET)
    payloads += [json.loads(m) for m in session.finish().messages]

    corrections = [p for p in payloads if p["type"] == "speakers"]
    assert corrections, "no second thoughts were ever sent"
    labels = {}
    for payload in corrections:
        labels.update(payload["labels"])
    finals = {str(p["sentence_id"]): p["speaker_id"]
              for p in payloads if p["type"] == "final"}
    for key, value in labels.items():
        finals[key] = value
    assert len(set(finals.values())) == 2, "the two voices stayed merged"


def test_nothing_is_sent_when_the_labels_were_already_right():
    session = merged_session()
    session.speaker_history.every = 1
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 20, LOUD)
    assert [p for p in payloads if p["type"] == "speakers"] == []


def test_a_session_without_a_speaker_model_has_nothing_to_recluster():
    assert make_session().speaker_history is None


# ---------------------------------------------------------------------------
# A stage that breaks
# ---------------------------------------------------------------------------
class BrokenClassifier:
    """An ordinary bug, repeated. Not a device failure - see CudaBroken."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, pcm: bytes):
        self.calls += 1
        raise ValueError("the classifier fell over")


def broken_noise_session():
    from server.pipeline.asr import Transcriber

    vad = ScriptedVAD((0.9,))
    return ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        noise_filter=NoiseFilter(classifier=BrokenClassifier()),
        transcriber=Transcriber(decoder=StubDecoder()),
    )


def test_a_broken_stage_does_not_eat_the_whole_meeting():
    """Reported from the pod: every sentence died in the noise filter, so two
    minutes of talking produced nothing at all - and the client had no way to
    tell that from nobody speaking."""
    session = broken_noise_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 40, LOUD)
    payloads += [json.loads(m) for m in session.finish().messages]
    finals = [p for p in payloads if p["type"] == "final"]
    assert finals, "the meeting produced nothing"
    assert finals[0]["transcript"]


def test_a_stage_that_keeps_breaking_is_switched_off():
    """One failure is a bad sentence. The same failure every time is a broken
    stage, and running it again per sentence only buries the evidence."""
    session = broken_noise_session()
    session.handle_text(Hello(session_id="abc").to_json())
    speak(session, 80, LOUD)
    session.finish()
    assert session.noise_filter is None


def test_the_client_is_told_which_stage_was_switched_off():
    session = broken_noise_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 80, LOUD)
    payloads += [json.loads(m) for m in session.finish().messages]
    errors = [p for p in payloads if p["type"] == "error"]
    assert errors, "the meeting went on short a stage and said nothing"
    assert "noise" in errors[0]["message"]
    assert errors[0]["fatal"] is False, "a missing stage is not fatal"


def test_a_stage_that_recovers_is_not_switched_off():
    """A single bad sentence must not cost the meeting a whole stage."""
    from server.pipeline.asr import Transcriber

    class SometimesBroken:
        def __init__(self):
            self.calls = 0

        def classify(self, pcm):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("one bad sentence")
            return SPEECHY

    vad = ScriptedVAD((0.9,))
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        noise_filter=NoiseFilter(classifier=SometimesBroken()),
        transcriber=Transcriber(decoder=StubDecoder()),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    speak(session, 80, LOUD)
    session.finish()
    assert session.noise_filter is not None


def test_the_failures_are_counted_for_the_summary():
    session = broken_noise_session()
    session.handle_text(Hello(session_id="abc").to_json())
    speak(session, 40, LOUD)
    session.finish()
    assert session.stats.stage_failures.get("noise", 0) >= 1


class CudaBroken:
    """What the pod did. The second call did not raise - it segfaulted."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, pcm: bytes):
        self.calls += 1
        raise RuntimeError(
            "cuDNN Frontend error: [cudnn_frontend] Error: "
            "No valid execution plans built")


def cuda_broken_session():
    from server.pipeline.asr import Transcriber

    vad = ScriptedVAD((0.9,))
    broken = CudaBroken()
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        noise_filter=NoiseFilter(classifier=broken),
        transcriber=Transcriber(decoder=StubDecoder()),
    )
    return session, broken


def test_a_cuda_failure_switches_the_stage_off_at_once():
    """The pod's timeline: the cuDNN error raised, then Whisper and ECAPA both
    kept working on CUDA for five more seconds, and the process died on the
    next call into the broken stage. Going back in is what killed it, so a
    three-strike rule is three chances to segfault."""
    session, broken = cuda_broken_session()
    session.handle_text(Hello(session_id="abc").to_json())
    speak(session, 80, LOUD)
    session.finish()
    assert session.noise_filter is None
    assert broken.calls == 1, "the broken CUDA path was entered again"


def test_a_cuda_failure_says_so_rather_than_blaming_the_stage():
    session, _ = cuda_broken_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 80, LOUD)
    payloads += [json.loads(m) for m in session.finish().messages]
    errors = [p for p in payloads if p["type"] == "error"]
    assert errors
    assert "CUDA" in errors[0]["message"]
    assert errors[0]["fatal"] is False


def test_the_meeting_carries_on_without_the_stage():
    session, _ = cuda_broken_session()
    session.handle_text(Hello(session_id="abc").to_json())
    payloads = speak(session, 80, LOUD)
    payloads += [json.loads(m) for m in session.finish().messages]
    assert [p for p in payloads if p["type"] == "final"]


def test_an_ordinary_failure_still_gets_its_three_chances():
    """A bad sentence is not a broken device."""
    session = broken_noise_session()
    session.handle_text(Hello(session_id="abc").to_json())
    speak(session, 40, LOUD)
    assert session.noise_filter is not None


# ---------------------------------------------------------------------------
# What the ASR is given
# ---------------------------------------------------------------------------
class RecordingResolver:
    """Stands in for the overlap resolver and remembers what it was handed."""

    def __init__(self) -> None:
        self.seen: list[bytes] = []

    def resolve(self, pcm: bytes):
        self.seen.append(pcm)
        from server.pipeline.overlap import Shaped
        return Shaped(pcm=b"\x01\x02" * (len(pcm) // 2), shaped=True,
                      level_dbfs=-20.0, gate_threshold_db=-30.0,
                      compressor_threshold_db=-10.0)


def test_the_committed_sentence_is_decoded_from_shaped_audio():
    from server.pipeline.asr import Transcriber

    decoder = StubDecoder()
    vad = ScriptedVAD([0.9] * 14 + [0.02])
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        overlap_resolver=RecordingResolver(),
        transcriber=Transcriber(decoder=decoder),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    speak_then_pause(session)
    finals = [call for call in decoder.calls if call[2] == 5]
    assert finals, "no committed sentence was decoded"


def test_the_running_text_is_decoded_from_raw_audio():
    """Not a detail: the gate removes quiet syllables, so the two decodes see
    different audio and can disagree on words the reader already read."""
    from server.pipeline.asr import Transcriber

    resolver = RecordingResolver()
    vad = ScriptedVAD((0.9,))
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        overlap_resolver=resolver,
        transcriber=Transcriber(decoder=StubDecoder()),
    )
    session.handle_text(Hello(session_id="abc").to_json())
    speak(session, 20, LOUD)
    assert session.stats.partials > 0
    assert resolver.seen == [], "the running text went through the resolver"


# ---------------------------------------------------------------------------
# Two languages in one utterance
# ---------------------------------------------------------------------------
class LidByLength:
    """Answers by how much audio it is given.

    That is the mechanism, not a convenience: the running text is decoded
    from the last few seconds, the committed sentence from the whole
    utterance, so the two see different audio and can land on different
    languages.
    """

    #: The running text here spans four chunks; the committed sentence is
    #: longer, because it also carries the pre-roll and the hangover.
    BOUNDARY_BYTES = CHUNK_BYTES * 4

    def __init__(self, short: str = "ja", long: str = "vi"):
        self.short = short
        self.long = long
        self.seen: list[int] = []

    def identify(self, pcm: bytes):
        from server.pipeline.lid import LanguageDecision
        self.seen.append(len(pcm))
        answer = self.long if len(pcm) > self.BOUNDARY_BYTES else self.short
        return LanguageDecision(lang_code=answer, confidence=0.9, margin=0.9,
                                reason="scripted")

    def reset(self) -> None:
        self.seen.clear()


def language_flip_session(identifier):
    from server.pipeline.asr import Transcriber

    vad = ScriptedVAD([0.9] * 14 + [0.02])
    return ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=vad),
        language_identifier=identifier,
        transcriber=Transcriber(decoder=StubDecoder()),
    )


def test_a_language_that_changes_between_prediction_and_sentence_is_counted():
    """Measured on a real meeting: the running text was Japanese twice over,
    the committed sentence came out Vietnamese, and the Japanese content was
    not mistranslated - it was gone. Two speakers, one utterance."""
    session = language_flip_session(LidByLength(short="ja", long="vi"))
    session.handle_text(Hello(session_id="abc").to_json())
    speak_then_pause(session)
    assert session.stats.language_flips == 1


def test_one_language_throughout_is_not_a_flip():
    session = language_flip_session(LidByLength(short="vi", long="vi"))
    session.handle_text(Hello(session_id="abc").to_json())
    speak_then_pause(session)
    assert session.stats.language_flips == 0


def test_an_utterance_with_no_prediction_before_it_is_not_a_flip():
    """Nothing to disagree with is not a disagreement."""
    session = language_flip_session(LidByLength(short="ja", long="vi"))
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    session.finish()
    assert session.stats.language_flips == 0
