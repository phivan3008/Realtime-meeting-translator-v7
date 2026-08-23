"""REAL END-TO-END TEST - Windows client streams live audio to the GPU server.

MUST RUN ON: the Windows Client PC, with the server already running on the
GPU pod and reachable from this machine.

This is the first test that crosses the machine boundary, so it proves the
seam rather than either side in isolation:

1. The pod is reachable and advertises the same audio contract we compiled
   against.
2. The handshake is accepted.
3. Live loopback audio flows continuously for the whole run, with nothing
   dropped by back-pressure.
4. The server hears real speech in it and pushes ``vad`` events back.
5. Those events arrive quickly enough to be useful: the lag between the
   moment an event's audio was captured and the moment the event lands back
   on the client is the true end-to-end latency of the pipeline so far.
6. The byte count the client sent matches the audio duration, i.e. no chunk
   was mangled on the way.

Usage
-----
    py -3.11 client\\tests_real\\test_real_stream.py --url ws://127.0.0.1:8000
    py -3.11 client\\tests_real\\test_real_stream.py --url ws://10.0.0.5:8000 --seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.audio.capture import (  # noqa: E402
    AudioCaptureError,
    LoopbackCapture,
)
from client.net.ws_client import StreamClient  # noqa: E402
from common.protocol import (  # noqa: E402
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
)

# Mirrors server/config.py; the client only needs it to sanity check what the
# server reports.
MAX_UTTERANCE_MS = 7_000
UTTERANCE_TOLERANCE_MS = 40

DEFAULT_URL = "ws://127.0.0.1:8000"
DEFAULT_SECONDS = 25.0

# An event must come back well within the time a person waits before deciding
# the subtitles are broken. The pre-roll alone accounts for ~256 ms of it.
MAX_EVENT_LAG_MS = 1_500.0
# Chunks the client is allowed to lose to back-pressure over the whole run.
MAX_DROPPED_CHUNKS = 0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f" - {detail}" if detail else ""))
        return passed

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


@dataclass
class Collected:
    """Server messages, stamped with the moment they reached the client."""

    stream_start: float = 0.0
    messages: list[tuple[float, dict]] = field(default_factory=list)

    def record(self, payload: dict) -> None:
        self.messages.append((time.perf_counter(), payload))

    @property
    def vad_events(self) -> list[tuple[float, dict]]:
        return [(t, m) for t, m in self.messages if m.get("type") == "vad"]

    @property
    def utterances(self) -> list[dict]:
        return [m for _, m in self.messages if m.get("type") == "utterance"]

    def lags_ms(self) -> list[float]:
        """How late each event arrived, relative to the audio it describes.

        ``at_ms`` is the event's position in the stream the server received,
        so the audio it refers to was captured at ``stream_start + at_ms``.
        Anything after that is capture buffering, network and server compute.
        """
        return [
            (arrived - self.stream_start) * 1000.0 - float(payload["at_ms"])
            for arrived, payload in self.vad_events
        ]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def check_health(base_url: str, report: Report) -> bool:
    """Ask /health before opening a socket, so a typo fails in one second."""
    http_url = base_url.replace("ws://", "http://").replace("wss://", "https://")
    health_url = http_url.rstrip("/") + "/health"
    print(f"\n  GET {health_url}")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        report.add("Server is reachable", False, f"{type(exc).__name__}: {exc}")
        return False

    print(f"  {payload}")
    report.add("Server is reachable", payload.get("status") == "ok",
               f"status={payload.get('status')!r}")
    report.add("Server has the VAD model loaded", bool(payload.get("vad_loaded")),
               f"vad_loaded={payload.get('vad_loaded')}")
    report.add(
        "Server has the noise filter loaded",
        bool(payload.get("noise_filter_loaded")),
        payload.get("noise_filter_error")
        or f"noise_filter_loaded={payload.get('noise_filter_loaded')}",
    )
    contract_ok = (
        payload.get("protocol_version") == PROTOCOL_VERSION
        and payload.get("sample_rate") == SAMPLE_RATE
        and payload.get("chunk_bytes") == CHUNK_BYTES
        and payload.get("chunk_ms") == CHUNK_DURATION_MS
    )
    report.add(
        "Client and server agree on the audio contract", contract_ok,
        f"server: v{payload.get('protocol_version')} "
        f"{payload.get('sample_rate')} Hz {payload.get('chunk_bytes')} B; "
        f"client: v{PROTOCOL_VERSION} {SAMPLE_RATE} Hz {CHUNK_BYTES} B",
    )
    return contract_ok


async def stream_for(url: str, device_hint: str | None, seconds: float,
                     report: Report) -> tuple[StreamClient, Collected]:
    collected = Collected()
    client = StreamClient(url.rstrip("/") + "/ws/stream",
                          on_message=collected.record)
    task = asyncio.create_task(client.run())

    if not await client.wait_connected(timeout=10.0):
        report.add("Handshake accepted", False,
                   "; ".join(client.stats.errors) or "timed out")
        await client.stop()
        await task
        return client, collected
    report.add("Handshake accepted", True, f"session {client.session_id}")

    capture = LoopbackCapture(device_name_hint=device_hint)
    device = capture.start()
    report.add("Loopback device opened", True, device.name)

    print(f"\n  Streaming {seconds:.0f} s - PLAY SPEECH NOW ...")
    collected.stream_start = time.perf_counter()
    deadline = collected.stream_start + seconds
    try:
        while time.perf_counter() < deadline:
            chunk = await asyncio.to_thread(capture.read, 0.5)
            if chunk is None:
                continue
            client.send(chunk)
            elapsed = time.perf_counter() - collected.stream_start
            print(f"\r  t={elapsed:5.1f}s  sent={client.stats.chunks_sent:<5d} "
                  f"queue={client.queued_chunks:<3d} "
                  f"dropped={client.stats.chunks_dropped}  "
                  f"events={len(collected.vad_events)}", end="", flush=True)
    finally:
        capture.stop()
        await client.stop(drain_timeout=2.0)
        await asyncio.wait_for(task, timeout=10.0)
    print()
    return client, collected


def check_stream(client: StreamClient, collected: Collected, seconds: float,
                 report: Report) -> None:
    stats = client.stats
    expected_chunks = seconds * 1000 / CHUNK_DURATION_MS
    print(f"\n  Sent {stats.chunks_sent} chunks "
          f"({stats.audio_seconds_sent:.1f} s of audio) in {seconds:.1f} s, "
          f"{stats.messages_received} messages back")

    report.add("Audio streamed for the whole run",
               stats.chunks_sent >= expected_chunks * 0.9,
               f"{stats.chunks_sent} chunks, expected about {expected_chunks:.0f}")
    report.add("Nothing dropped by back-pressure",
               stats.chunks_dropped <= MAX_DROPPED_CHUNKS,
               f"dropped {stats.chunks_dropped}")
    report.add("Audio duration matches the bytes sent",
               abs(stats.audio_seconds_sent - seconds) < seconds * 0.15,
               f"{stats.audio_seconds_sent:.1f} s of audio vs {seconds:.1f} s wall")
    report.add("The connection survived the run", stats.connects == 1,
               f"{stats.connects} connect(s), {stats.disconnects} disconnect(s)")
    report.add("No client-side protocol errors", not stats.errors,
               "; ".join(stats.errors) or "none")


def check_events(collected: Collected, report: Report) -> None:
    events = collected.vad_events
    print("\n  VAD events from the server:")
    for arrived, payload in events:
        lag = (arrived - collected.stream_start) * 1000.0 - float(payload["at_ms"])
        print(f"    at_ms={float(payload['at_ms']):8.1f}  {payload['event']:<12} "
              f"arrived +{lag:6.0f} ms later")
    if not events:
        print("    (none)")

    report.add("The server heard speech", len(events) >= 1,
               f"{len(events)} vad event(s)")
    starts = sum(1 for _, m in events if m["event"] == "speech_start")
    ends = sum(1 for _, m in events if m["event"] == "speech_end")
    # Exact balance, not off-by-one: the client stops with a `bye`, and the
    # server closes any segment still open before the socket goes away. An
    # unmatched start means the last sentence would never be finalised.
    report.add("Every speech segment is opened and closed", starts == ends,
               f"{starts} speech_start, {ends} speech_end")

    lags = collected.lags_ms()
    if lags:
        worst = max(lags)
        print(f"\n  End-to-end lag: mean {statistics.fmean(lags):.0f} ms, "
              f"max {worst:.0f} ms (budget {MAX_EVENT_LAG_MS:.0f} ms)")
        report.add("Events come back fast enough to be useful",
                   worst < MAX_EVENT_LAG_MS,
                   f"worst {worst:.0f} ms < {MAX_EVENT_LAG_MS:.0f} ms")
        report.add("No event arrives before its audio was captured",
                   min(lags) > -CHUNK_DURATION_MS,
                   f"smallest lag {min(lags):.0f} ms")


def check_utterances(collected: Collected, report: Report) -> None:
    """The sentence boundaries the Stream Buffer Manager committed."""
    utterances = collected.utterances
    print()
    print("  Utterances committed by the server:")
    for payload in utterances:
        verdict = "keep" if payload.get("kept", True) else f"DROP {payload.get('label', '')}"
        print(f"    #{payload['index']:<3} "
              f"{payload['start_ms'] / 1000:7.2f}s -> "
              f"{payload['end_ms'] / 1000:7.2f}s "
              f"({payload['duration_ms'] / 1000:.2f}s)  "
              f"{payload['reason']:<14} "
              f"speech={payload.get('speech_score', 0):.2f}  {verdict}"
              + ("  [continues]" if payload["continues_previous"] else ""))
    if not utterances:
        print("    (none)")

    report.add("The server committed at least one sentence",
               len(utterances) >= 1, f"{len(utterances)} utterance(s)")
    if not utterances:
        return

    report.add(
        "Utterance indexes are consecutive from zero",
        [u["index"] for u in utterances] == list(range(len(utterances))),
        f"indexes {[u['index'] for u in utterances][:6]}",
    )
    longest = max(u["duration_ms"] for u in utterances)
    report.add(
        "No sentence outstays the max duration",
        longest <= MAX_UTTERANCE_MS + UTTERANCE_TOLERANCE_MS,
        f"longest {longest / 1000:.2f} s, limit {MAX_UTTERANCE_MS / 1000:.1f} s",
    )
    overlaps = [
        (a["index"], b["index"])
        for a, b in zip(utterances, utterances[1:])
        if b["start_ms"] < a["end_ms"] - 1e-6
    ]
    report.add("Sentences never overlap", not overlaps, f"{overlaps[:3]}")
    report.add(
        "A continued sentence joins the previous one with no gap",
        all(abs(b["start_ms"] - a["end_ms"]) < 1.0
            for a, b in zip(utterances, utterances[1:])
            if b["continues_previous"]),
        f"{sum(1 for u in utterances if u['continues_previous'])} continuation(s)",
    )

    ends = sum(1 for _, m in collected.vad_events if m["event"] == "speech_end")
    report.add(
        "Every closed speech segment produced at least one sentence",
        len(utterances) >= ends,
        f"{len(utterances)} utterance(s) for {ends} speech_end",
    )

    kept = [u for u in utterances if u.get("kept", True)]
    dropped = [u for u in utterances if not u.get("kept", True)]
    report.add(
        "The noise filter did not eat the whole meeting",
        len(kept) >= 1,
        f"{len(kept)} kept, {len(dropped)} dropped as noise",
    )
    # Whether the scores are any good is the server-side noise test's job;
    # here we only prove every sentence went past the filter and carries its
    # verdict.
    report.add(
        "Every sentence carries the filter's verdict",
        all("kept" in u and "speech_score" in u for u in utterances),
        f"{len(utterances)} sentence(s)",
    )
    report.add(
        "Every dropped sentence says what it sounded like",
        all(u.get("label") for u in dropped),
        f"{[u.get('label') for u in dropped][:3]}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main_async(args) -> int:
    print("=" * 72)
    print("REAL TEST - CLIENT TO SERVER AUDIO STREAM")
    print(f"target: {args.url}   {SAMPLE_RATE} Hz mono, "
          f"{CHUNK_DURATION_MS} ms chunks ({CHUNK_BYTES} bytes)")
    print("=" * 72)

    report = Report()
    print("\nChecks:")
    if not check_health(args.url, report):
        print("\nRESULT: FAIL - server not usable, stopping before streaming")
        return 2

    client, collected = await stream_for(args.url, args.device, args.seconds,
                                         report)
    if client.stats.connects == 0 or not client.stats.chunks_sent:
        print("\nRESULT: FAIL - never streamed anything")
        for error in client.stats.errors:
            print(f"  - {error}")
        return 1

    check_stream(client, collected, args.seconds, report)
    check_events(collected, report)
    check_utterances(collected, report)

    print("\n" + "=" * 72)
    if report.failed:
        print(f"RESULT: FAIL ({len(report.failed)} check(s) failed)")
        for check in report.failed:
            print(f"  - {check.name}: {check.detail}")
        print("=" * 72)
        return 1
    print(f"RESULT: PASS ({len(report.checks)} checks)")
    print("=" * 72)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"server base URL (default: {DEFAULT_URL})")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--device", default=None,
                        help="substring of the loopback device name to use")
    args = parser.parse_args()

    try:
        return asyncio.run(main_async(args))
    except AudioCaptureError as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
