"""WebSocket tests for the FastAPI app, driven by Starlette's TestClient.

These exercise the real routing, the real handshake and the real teardown -
only the Silero model is stubbed, so no torch and no GPU are needed.

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


def wait_slot_free(timeout: float = 30.0) -> bool:
    """The server releases the slot on its own task, so give it a moment.

    Closing the client end of a TestClient websocket does not synchronously
    join the server coroutine, so asserting on the shared state immediately
    after the ``with`` block is a race.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app_module.state.active_session_id is None:
            return True
        time.sleep(0.001)
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
STAGE_CLASSES = ("NoiseFilter", "OverlapResolver", "SpeakerIdentifier",
                 "LanguageIdentifier", "Transcriber", "Translator")


def stub_stages(monkeypatch, **overrides):
    monkeypatch.setattr(app_module, "SileroVAD", lambda: object())
    monkeypatch.setattr(app_module, "AstClassifier", lambda: object())
    for name in STAGE_CLASSES:
        monkeypatch.setattr(app_module, name,
                            overrides.get(name, lambda *a, **k: object()))


def test_the_noise_filter_is_off_unless_asked_for(monkeypatch):
    """Measured over two real meetings: a quarter of the socket thread, and
    nothing dropped. The old code also returned out of load_models() when it
    was switched off, leaving the pod without an ASR either."""
    monkeypatch.setattr(app_module.config, "ENABLE_NOISE_FILTER", False)
    stub_stages(monkeypatch)
    state = app_module.AppState()
    state.load_models()

    assert state.noise_filter is None
    assert "ENABLE_NOISE_FILTER" in state.noise_error
    for attribute in ("overlap_resolver", "speaker_identifier",
                      "language_identifier", "transcriber", "translator"):
        assert getattr(state, attribute) is not None, attribute


def test_the_noise_filter_loads_when_asked_for(monkeypatch):
    monkeypatch.setattr(app_module.config, "ENABLE_NOISE_FILTER", True)
    stub_stages(monkeypatch)
    state = app_module.AppState()
    state.load_models()
    assert state.noise_filter is not None
    assert state.noise_error == ""


def test_the_noise_filter_runs_on_the_cpu_by_default(monkeypatch):
    """On the pod AST on CUDA raised in cuDNN and the next call segfaulted."""
    import importlib

    from server import config

    monkeypatch.delenv("NOISE_DEVICE", raising=False)
    try:
        assert importlib.reload(config).NOISE_DEVICE == "cpu"
    finally:
        importlib.reload(config)


def test_one_stage_failing_does_not_stop_the_others(monkeypatch):
    from server.pipeline.asr import AsrError

    def explode():
        raise AsrError("no CUDA")

    monkeypatch.setattr(app_module.config, "ENABLE_NOISE_FILTER", True)
    stub_stages(monkeypatch, Transcriber=explode)
    state = app_module.AppState()
    state.load_models()

    assert state.transcriber is None
    assert "no CUDA" in state.asr_error
    assert state.translator is not None, "a later stage was skipped"


def test_a_stage_that_failed_is_not_retried(monkeypatch):
    """Reconnecting must not spend thirty seconds retrying a missing model."""
    from server.pipeline.asr import AsrError
    attempts = []

    def explode():
        attempts.append(1)
        raise AsrError("no CUDA")

    stub_stages(monkeypatch, Transcriber=explode)
    state = app_module.AppState()
    state.load_models()
    state.load_models()
    assert len(attempts) == 1


def test_a_failed_overlap_resolver_says_why(monkeypatch):
    """It once failed with no reason recorded anywhere, and an A/B of the
    stage ran twice without it."""
    from server.pipeline.overlap import OverlapError

    def explode():
        raise OverlapError("pedalboard is not installed.")

    monkeypatch.setattr(app_module.config, "DISABLE_OVERLAP", False)
    stub_stages(monkeypatch, OverlapResolver=explode)
    monkeypatch.setattr(app_module, "state", app_module.AppState())
    app_module.state.load_models()
    payload = app_module.health()
    assert payload["overlap_resolver_loaded"] is False
    assert payload["overlap_error"] == "pedalboard is not installed."


def test_the_overlap_resolver_can_be_turned_off(monkeypatch):
    """Its only consumer is the ASR, so this is the switch for feeding
    Whisper raw audio."""
    monkeypatch.setattr(app_module.config, "DISABLE_OVERLAP", True)
    stub_stages(monkeypatch)
    state = app_module.AppState()
    state.load_models()
    assert state.overlap_resolver is None
    assert state.overlap_error == "disabled by DISABLE_OVERLAP"
    assert state.transcriber is not None, "it took the ASR down with it"


def test_the_slot_wait_returns_at_once_when_it_is_already_free():
    app_module.state.active_session_id = None
    started = time.monotonic()
    assert wait_slot_free() is True
    assert time.monotonic() - started < 0.1


def test_a_connection_dropped_mid_meeting_frees_the_slot(client):
    """The receive loop raises on a dropped socket and never returns a flag,
    so the slot is released by id."""
    loud(client, [0.9])
    with client.websocket_connect("/ws/stream") as socket:
        socket.send_text(Hello(session_id="gone").to_json())
        socket.receive_text()
        for _ in range(3):
            socket.send_bytes(chunk())
    assert wait_slot_free()


# ---------------------------------------------------------------------------
# Which interpreter, and which settings
# ---------------------------------------------------------------------------
def test_health_names_the_interpreter():
    """Running from the conda base interpreter is silent - it starts, loads
    most of the pipeline and serves meetings. It cost this project several
    measurements taken against a pipeline that was not the one under test."""
    payload = app_module.health()
    assert payload["python"]
    assert isinstance(payload["in_venv"], bool)


def test_a_virtual_environment_is_recognised(monkeypatch):
    monkeypatch.setattr(app_module.sys, "prefix", "/workspace/project/.venv")
    monkeypatch.setattr(app_module.sys, "base_prefix", "/usr")
    assert app_module.in_venv() is True
    assert "NOT a venv" not in app_module.which_environment()


def test_the_conda_base_interpreter_is_not_mistaken_for_one(monkeypatch):
    monkeypatch.setattr(app_module.sys, "prefix", "/opt/conda")
    monkeypatch.setattr(app_module.sys, "base_prefix", "/opt/conda")
    assert app_module.in_venv() is False
    assert "NOT a venv" in app_module.which_environment()


def test_the_startup_log_says_which_interpreter(monkeypatch, caplog):
    stub_stages(monkeypatch)
    with caplog.at_level(logging.INFO):
        app_module.AppState().load_models()
    assert "Python:" in caplog.text


def test_health_reports_environment_overrides(monkeypatch):
    """A variable left over from an earlier terminal changes what the
    pipeline does and says nothing."""
    monkeypatch.setenv("LANGUAGE_SPLIT", "0")
    assert app_module.health()["overrides"]["LANGUAGE_SPLIT"] == "0"


def test_no_overrides_reads_as_empty(monkeypatch):
    from server.config import known_variables

    for name in known_variables():
        monkeypatch.delenv(name, raising=False)
    assert app_module.health()["overrides"] == {}


def test_every_variable_the_config_reads_is_known():
    """The list is derived from config.py, so a new one cannot be forgotten."""
    import re

    from server.config import known_variables

    source = (Path(app_module.__file__).parent / "config.py").read_text(
        encoding="utf-8")
    found = set(re.findall(r'os\.environ\.get\(\s*"(\w+)"', source))
    found |= set(re.findall(r'_flag\(\s*"(\w+)"', source))
    assert set(known_variables()) == found
    assert {"ENABLE_NOISE_FILTER", "DISABLE_OVERLAP", "LANGUAGE_SPLIT",
            "ASR_PROMPT_ON_PARTIALS", "MEETING_DATA_DIR"} <= found


def test_the_startup_log_says_when_nothing_is_overridden(monkeypatch, caplog):
    from server.config import known_variables

    for name in known_variables():
        monkeypatch.delenv(name, raising=False)
    stub_stages(monkeypatch)
    with caplog.at_level(logging.INFO):
        app_module.AppState().load_models()
    assert "No environment overrides" in caplog.text


def test_no_server_code_reads_the_environment_behind_the_configs_back():
    """Only config.py may read a variable, or the override report misses it
    - which is what happened to ENABLE_NOISE_FILTER on another branch."""
    import re

    server = Path(app_module.__file__).parent
    for path in server.rglob("*.py"):
        if {"tests", "tests_real", "analysis"} & set(path.parts):
            continue
        if path.name in ("config.py", "launch_vllm.py"):
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"os\.environ", text), path


# ---------------------------------------------------------------------------
# The end-of-meeting summary
# ---------------------------------------------------------------------------
def test_the_summary_reports_what_was_carried_over(caplog):
    """Splits, stage failures and speakers each get a line, because each of
    them has been a silent failure once."""
    from server.net.session import ServerSession
    from server.pipeline.language_split import LanguageSplitter
    from server.pipeline.reclustering import SpeakerHistory
    from server.pipeline.vad import VADSegmenter

    session = ServerSession(segmenter_factory=lambda: VADSegmenter(
        vad=ScriptedVAD([0.02])))
    session.language_splitter = LanguageSplitter(prober=None)
    session.speaker_history = SpeakerHistory()
    session.speaker_history.stats.stop_scores = [0.1, 0.2, 0.25]
    session.stats.stage_failures = {"noise": 1}
    with caplog.at_level(logging.INFO):
        app_module._log_summary(session)
    assert "language splits" in caplog.text
    assert "stages that raised" in caplog.text
    assert "refused merges [0.1" in caplog.text
