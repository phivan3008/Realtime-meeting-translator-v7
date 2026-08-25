"""WebSocket tests for the FastAPI app, driven by Starlette's TestClient.

These exercise the real routing, the real handshake and the real teardown -
only the Silero model is stubbed, so no torch and no GPU are needed.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.protocol import CHUNK_BYTES, Hello, make_bye  # noqa: E402
from server import app as app_module  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES, VADSegmenter  # noqa: E402


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


@pytest.fixture
def client(monkeypatch):
    """A TestClient whose VAD is scripted rather than loaded from disk."""
    vad = ScriptedVAD([0.02])
    monkeypatch.setattr(app_module.state, "vad", vad)
    monkeypatch.setattr(app_module.state, "active_session_id", None)
    monkeypatch.setattr(app_module.state, "load_models", lambda: None)
    monkeypatch.setattr(
        app_module.state, "make_segmenter", lambda: VADSegmenter(vad=vad)
    )
    with TestClient(app_module.app) as test_client:
        test_client.scripted_vad = vad
        yield test_client


def loud(client, probabilities) -> None:
    """Rewrite the scripted probabilities before opening a connection."""
    client.scripted_vad.script = list(probabilities)
    client.scripted_vad.calls = 0


def chunk(value: int = 1000) -> bytes:
    return np.full(CHUNK_BYTES // 2, value, dtype="<i2").tobytes()


def wait_slot_free(timeout: float = 2.0) -> bool:
    """The server releases the slot on its own task, so give it a moment.

    Closing the client end of a TestClient websocket does not synchronously
    join the server coroutine, so asserting on the shared state immediately
    after the ``with`` block is a race.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app_module.state.active_session_id is None:
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_reports_the_audio_contract(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["chunk_bytes"] == CHUNK_BYTES
    assert payload["sample_rate"] == 16_000
    assert payload["session_active"] is False


# ---------------------------------------------------------------------------
# Handshake over a real socket
# ---------------------------------------------------------------------------
def test_hello_is_answered_with_ready(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="abc").to_json())
        payload = json.loads(socket.receive_text())
    assert payload["type"] == "ready"
    assert payload["session_id"] == "abc"


def test_a_bad_audio_format_is_refused_and_the_socket_closes(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="abc", sample_rate=8_000).to_json())
        payload = json.loads(socket.receive_text())
        assert payload["type"] == "error"
        assert "sample_rate" in payload["message"]
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()


def test_audio_sent_before_hello_is_refused(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_bytes(chunk())
        payload = json.loads(socket.receive_text())
    assert payload["type"] == "error"
    assert "before a valid hello" in payload["message"]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def test_silence_streams_without_producing_any_message(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="abc").to_json())
        json.loads(socket.receive_text())               # ready
        for _ in range(5):
            socket.send_bytes(chunk())
        socket.send_text(make_bye("done"))
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()


def test_speech_pushes_vad_events_back_to_the_client(client):
    loud(client, [0.9])
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="abc").to_json())
        json.loads(socket.receive_text())               # ready
        socket.send_bytes(chunk())
        payload = json.loads(socket.receive_text())
    assert payload["type"] == "vad"
    assert payload["event"] == "speech_start"
    assert payload["at_ms"] == 0.0


def test_a_dropped_connection_still_closes_the_open_segment(client):
    """The segment must not stay open just because the client vanished."""
    loud(client, [0.9])
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="abc").to_json())
        socket.receive_text()                           # ready
        socket.send_bytes(chunk())
        assert json.loads(socket.receive_text())["event"] == "speech_start"
    # Leaving the context closes the socket; the server runs finish().
    assert wait_slot_free()


def test_a_wrong_sized_chunk_is_refused_mid_stream(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="abc").to_json())
        socket.receive_text()                           # ready
        socket.send_bytes(bytes(1234))
        payload = json.loads(socket.receive_text())
    assert payload["type"] == "error"
    assert "exactly 6400 bytes" in payload["message"]


# ---------------------------------------------------------------------------
# One meeting at a time
# ---------------------------------------------------------------------------
def test_a_second_concurrent_session_is_refused(client):
    with client.websocket_connect("/ws/stream") as first:
        first.send_text(Hello(session_id="first").to_json())
        first.receive_text()                            # ready
        first.send_bytes(chunk())

        with client.websocket_connect("/ws/stream") as second:
            payload = json.loads(second.receive_text())
        assert payload["type"] == "error"
        assert "one meeting at a time" in payload["message"]


def test_the_slot_is_released_when_the_first_session_ends(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="first").to_json())
        socket.receive_text()
        socket.send_bytes(chunk())
    assert wait_slot_free()

    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="second").to_json())
        payload = json.loads(socket.receive_text())
    assert payload["type"] == "ready"
    assert payload["session_id"] == "second"


def test_a_refused_handshake_does_not_hold_the_slot(client):
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="bad", channels=2).to_json())
        assert json.loads(socket.receive_text())["type"] == "error"
    assert wait_slot_free()

    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="good").to_json())
        assert json.loads(socket.receive_text())["type"] == "ready"


# ---------------------------------------------------------------------------
# Loading the stages
# ---------------------------------------------------------------------------
def test_disabling_the_noise_filter_still_loads_everything_else(monkeypatch):
    """The old code returned out of load_models() here, so setting
    DISABLE_NOISE_FILTER silently left the pod without an ASR either."""
    from server.app import AppState

    state = AppState()
    monkeypatch.setenv("DISABLE_NOISE_FILTER", "1")
    monkeypatch.setattr(app_module, "SileroVAD", lambda: object())
    for name in ("OverlapResolver", "SpeakerIdentifier", "LanguageIdentifier",
                 "Transcriber", "Translator"):
        monkeypatch.setattr(app_module, name, lambda: object())

    state.load_models()

    assert state.noise_filter is None
    assert state.noise_error == "disabled by DISABLE_NOISE_FILTER"
    assert state.overlap_resolver is not None
    assert state.speaker_identifier is not None
    assert state.language_identifier is not None
    assert state.transcriber is not None
    assert state.translator is not None


def test_one_stage_failing_does_not_stop_the_others(monkeypatch):
    from server.app import AppState
    from server.pipeline.asr import AsrError

    state = AppState()
    monkeypatch.delenv("DISABLE_NOISE_FILTER", raising=False)
    monkeypatch.setattr(app_module, "SileroVAD", lambda: object())

    def explode():
        raise AsrError("no CUDA")

    monkeypatch.setattr(app_module, "Transcriber", explode)
    for name in ("NoiseFilter", "OverlapResolver", "SpeakerIdentifier",
                 "LanguageIdentifier", "Translator"):
        monkeypatch.setattr(app_module, name, lambda *a, **k: object())
    monkeypatch.setattr(app_module, "AstClassifier", lambda: object())

    state.load_models()

    assert state.transcriber is None
    assert "no CUDA" in state.asr_error
    assert state.translator is not None, "a later stage was skipped"


def test_a_stage_that_failed_is_not_retried(monkeypatch):
    """Reconnecting must not spend thirty seconds retrying a missing model."""
    from server.app import AppState
    from server.pipeline.asr import AsrError

    state = AppState()
    monkeypatch.delenv("DISABLE_NOISE_FILTER", raising=False)
    monkeypatch.setattr(app_module, "SileroVAD", lambda: object())
    attempts = []

    def explode():
        attempts.append(1)
        raise AsrError("no CUDA")

    monkeypatch.setattr(app_module, "Transcriber", explode)
    for name in ("NoiseFilter", "OverlapResolver", "SpeakerIdentifier",
                 "LanguageIdentifier", "Translator"):
        monkeypatch.setattr(app_module, name, lambda *a, **k: object())
    monkeypatch.setattr(app_module, "AstClassifier", lambda: object())

    state.load_models()
    state.load_models()
    assert len(attempts) == 1
