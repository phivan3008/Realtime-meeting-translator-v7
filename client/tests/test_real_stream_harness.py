"""Smoke tests for the ``client/tests_real/test_real_stream.py`` harness.

The audio capture part cannot run here - the Dev PC has no loopback device -
but everything around it can: the health probe, the lag arithmetic and the
reporting. Those are exactly the parts that would otherwise fail on the
Windows Client PC and cost a round trip.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.protocol import (  # noqa: E402
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
)


def load_harness():
    path = ROOT / "client" / "tests_real" / "test_real_stream.py"
    spec = importlib.util.spec_from_file_location("real_stream_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


GOOD_HEALTH = {
    "status": "ok",
    "protocol_version": PROTOCOL_VERSION,
    "sample_rate": SAMPLE_RATE,
    "chunk_bytes": CHUNK_BYTES,
    "chunk_ms": CHUNK_DURATION_MS,
    "vad_loaded": True,
    "session_active": False,
}


def serve_health(payload: dict | None):
    """Run a one-route HTTP server on a random port; returns its base URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                  # noqa: N802
            if self.path != "/health" or payload is None:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):                     # keep pytest quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"ws://127.0.0.1:{server.server_port}"


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------
def test_a_matching_server_passes_every_health_check():
    server, url = serve_health(GOOD_HEALTH)
    try:
        report = harness.Report()
        assert harness.check_health(url, report) is True
        assert report.failed == []
    finally:
        server.shutdown()


def test_a_mismatched_contract_is_caught_before_any_audio_is_sent():
    server, url = serve_health({**GOOD_HEALTH, "sample_rate": 48_000})
    try:
        report = harness.Report()
        assert harness.check_health(url, report) is False
        assert [c.name for c in report.failed] == [
            "Client and server agree on the audio contract"
        ]
    finally:
        server.shutdown()


def test_a_server_without_the_model_is_flagged():
    server, url = serve_health({**GOOD_HEALTH, "vad_loaded": False})
    try:
        report = harness.Report()
        harness.check_health(url, report)
        assert [c.name for c in report.failed] == [
            "Server has the VAD model loaded"
        ]
    finally:
        server.shutdown()


def test_an_unreachable_server_fails_without_raising():
    report = harness.Report()
    assert harness.check_health("ws://127.0.0.1:9", report) is False
    assert [c.name for c in report.failed] == ["Server is reachable"]


def test_the_https_scheme_is_translated_for_the_health_probe():
    server, url = serve_health(GOOD_HEALTH)
    try:
        report = harness.Report()
        assert harness.check_health(url.replace("ws://", "wss://"), report) is False
        # wss -> https against a plain HTTP server fails at the transport layer,
        # which is the point: the scheme is rewritten, not ignored.
        assert [c.name for c in report.failed] == ["Server is reachable"]
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Lag arithmetic
# ---------------------------------------------------------------------------
def make_collected(stream_start: float, events: list[tuple[float, float, str]]):
    """events = (arrival_perf_counter, at_ms, kind)."""
    collected = harness.Collected(stream_start=stream_start)
    for arrived, at_ms, kind in events:
        collected.messages.append(
            (arrived, {"type": "vad", "event": kind, "at_ms": at_ms})
        )
    return collected


def test_lag_is_the_delay_between_capture_and_arrival():
    # Audio at 1000 ms into the stream, event landed 1.4 s after start.
    collected = make_collected(100.0, [(101.4, 1000.0, "speech_start")])
    assert collected.lags_ms() == pytest.approx([400.0])


def test_only_vad_messages_count_towards_lag():
    collected = harness.Collected(stream_start=0.0)
    collected.messages.append((0.5, {"type": "ready", "session_id": "x"}))
    collected.messages.append((1.0, {"type": "vad", "event": "speech_start",
                                     "at_ms": 500.0}))
    assert len(collected.vad_events) == 1
    assert collected.lags_ms() == pytest.approx([500.0])


def test_record_stamps_each_message_on_arrival():
    collected = harness.Collected()
    collected.record({"type": "ready"})
    collected.record({"type": "vad", "event": "speech_end", "at_ms": 0})
    assert len(collected.messages) == 2
    assert collected.messages[0][0] <= collected.messages[1][0]


# ---------------------------------------------------------------------------
# Event checks
# ---------------------------------------------------------------------------
def test_a_healthy_event_stream_passes():
    collected = make_collected(
        0.0,
        [(0.6, 200.0, "speech_start"), (5.4, 5000.0, "speech_end")],
    )
    report = harness.Report()
    harness.check_events(collected, report)
    assert report.failed == []


def test_no_events_at_all_fails():
    report = harness.Report()
    harness.check_events(harness.Collected(stream_start=0.0), report)
    assert [c.name for c in report.failed] == ["The server heard speech"]


def test_an_event_that_arrives_too_late_fails():
    collected = make_collected(0.0, [(3.0, 200.0, "speech_start")])   # 2.8 s lag
    report = harness.Report()
    harness.check_events(collected, report)
    assert "Events come back fast enough to be useful" in [
        c.name for c in report.failed
    ]


def test_an_event_that_predates_its_own_audio_fails():
    """Impossible unless the timestamps are wrong somewhere."""
    collected = make_collected(0.0, [(0.1, 5000.0, "speech_start")])
    report = harness.Report()
    harness.check_events(collected, report)
    assert "No event arrives before its audio was captured" in [
        c.name for c in report.failed
    ]


def test_an_unclosed_final_segment_fails():
    """A start with no end means the last sentence never gets finalised."""
    collected = make_collected(
        0.0,
        [(0.3, 100.0, "speech_start"), (0.6, 200.0, "speech_end"),
         (0.9, 300.0, "speech_start")],
    )
    report = harness.Report()
    harness.check_events(collected, report)
    assert "Every speech segment is opened and closed" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# Stream checks
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, **overrides):
        from client.net.ws_client import ClientStats

        self.stats = ClientStats(
            chunks_sent=overrides.get("chunks_sent", 100),
            chunks_dropped=overrides.get("chunks_dropped", 0),
            bytes_sent=overrides.get("chunks_sent", 100) * CHUNK_BYTES,
            connects=overrides.get("connects", 1),
            disconnects=overrides.get("disconnects", 1),
            errors=overrides.get("errors", []),
        )


def test_a_clean_20_second_run_passes():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=100), harness.Collected(),
                         20.0, report)
    assert report.failed == []


def test_a_short_run_is_caught():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=40), harness.Collected(),
                         20.0, report)
    names = [c.name for c in report.failed]
    assert "Audio streamed for the whole run" in names
    assert "Audio duration matches the bytes sent" in names


def test_dropped_chunks_are_caught():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=100, chunks_dropped=3),
                         harness.Collected(), 20.0, report)
    assert [c.name for c in report.failed] == ["Nothing dropped by back-pressure"]


def test_a_reconnect_during_the_run_is_reported():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=100, connects=2),
                         harness.Collected(), 20.0, report)
    assert [c.name for c in report.failed] == ["The connection survived the run"]
