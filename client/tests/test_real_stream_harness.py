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
    "noise_filter_loaded": True,
    "noise_filter_error": "",
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


# ---------------------------------------------------------------------------
# Utterance checks
# ---------------------------------------------------------------------------
def utterance(index: int, start_ms: float, end_ms: float,
              reason: str = "pause", continues: bool = False,
              kept: bool = True, label: str = "",
              speech_score: float = 0.8) -> dict:
    return {
        "type": "utterance",
        "index": index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "reason": reason,
        "continues_previous": continues,
        "kept": kept,
        "label": label,
        "speech_score": speech_score,
    }


def with_utterances(*payloads, vad_ends: int = 0):
    collected = harness.Collected(stream_start=0.0)
    for i in range(vad_ends):
        collected.messages.append(
            (0.1, {"type": "vad", "event": "speech_end", "at_ms": float(i)})
        )
    for payload in payloads:
        collected.messages.append((0.1, payload))
    return collected


def test_a_healthy_set_of_utterances_passes():
    collected = with_utterances(
        utterance(0, 0, 2_000),
        utterance(1, 5_000, 6_000),
        vad_ends=2,
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert report.failed == []


def test_no_utterances_at_all_fails():
    report = harness.Report()
    harness.check_utterances(harness.Collected(stream_start=0.0), report)
    assert [c.name for c in report.failed] == [
        "The server committed at least one sentence"
    ]


def test_a_gap_in_the_indexes_is_caught():
    collected = with_utterances(utterance(0, 0, 100), utterance(2, 200, 300))
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Utterance indexes are consecutive from zero" in [
        c.name for c in report.failed
    ]


def test_a_sentence_longer_than_the_limit_is_caught():
    collected = with_utterances(utterance(0, 0, 9_000))
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "No sentence outstays the max duration" in [
        c.name for c in report.failed
    ]


def test_overlapping_sentences_are_caught():
    collected = with_utterances(utterance(0, 0, 2_000), utterance(1, 1_000, 3_000))
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Sentences never overlap" in [c.name for c in report.failed]


def test_a_continuation_with_a_gap_is_caught():
    collected = with_utterances(
        utterance(0, 0, 2_000),
        utterance(1, 4_000, 5_000, "max_duration", continues=True),
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "A continued sentence joins the previous one with no gap" in [
        c.name for c in report.failed
    ]


def test_a_closed_segment_with_no_sentence_is_caught():
    collected = with_utterances(utterance(0, 0, 1_000), vad_ends=3)
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Every closed speech segment produced at least one sentence" in [
        c.name for c in report.failed
    ]


def test_a_server_without_the_noise_filter_is_flagged():
    server, url = serve_health({
        **GOOD_HEALTH,
        "noise_filter_loaded": False,
        "noise_filter_error": "Could not load YAMNet from 'https://...'",
    })
    try:
        report = harness.Report()
        harness.check_health(url, report)
        assert [c.name for c in report.failed] == [
            "Server has the noise filter loaded"
        ]
    finally:
        server.shutdown()


def test_a_dropped_sentence_without_a_label_is_caught():
    collected = with_utterances(
        utterance(0, 0, 1_000),
        utterance(1, 2_000, 3_000, kept=False, label="", speech_score=0.01),
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Every dropped sentence says what it sounded like" in [
        c.name for c in report.failed
    ]


def test_a_filter_that_drops_everything_is_caught():
    collected = with_utterances(
        utterance(0, 0, 1_000, kept=False, label="Typing", speech_score=0.01),
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "The noise filter did not eat the whole meeting" in [
        c.name for c in report.failed
    ]
