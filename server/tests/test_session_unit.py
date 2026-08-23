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
    assert session.stats.events_sent == 3       # start, end, start


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
    payloads = [json.loads(m) for m in response.messages]
    assert [p["event"] for p in payloads] == [VADEvent.SPEECH_END.value]
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
        events += [json.loads(m) for m in session.handle_binary(chunk()).messages]
    events += [json.loads(m) for m in session.handle_text(make_bye("done")).messages]
    kinds = [e["event"] for e in events]
    assert kinds.count("speech_start") == kinds.count("speech_end") == 1


def test_finish_closes_a_segment_left_open_by_a_dropped_connection():
    """Otherwise the buffer manager waits forever for an end that never comes."""
    session = make_session([0.9])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    response = session.finish()
    payloads = [json.loads(m) for m in response.messages]
    assert [p["event"] for p in payloads] == [VADEvent.SPEECH_END.value]
    assert session.state is SessionState.CLOSED


def test_finish_on_a_silent_session_says_nothing():
    session = make_session([0.02])
    session.handle_text(Hello(session_id="abc").to_json())
    session.handle_binary(chunk())
    assert session.finish().messages == []


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
