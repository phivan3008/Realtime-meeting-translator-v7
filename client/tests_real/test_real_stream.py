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
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.audio.capture import (  # noqa: E402
    AudioCaptureError,
    LoopbackCapture,
)
from client.net.ws_client import StreamClient  # noqa: E402
from server.wordlists import Hallucinations  # noqa: E402
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

#: Audio this machine failed to capture, past which the lag figures stop
#: meaning anything: at_ms counts what the server received, so audio never
#: recorded here shifts every event's apparent lag by the same amount. A run
#: legitimately ends part-way through a chunk, so a little slack. Thirteen
#: seconds is not slack.
MISSING_AUDIO_TOLERANCE_S = 1.0


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

    @property
    def partials(self) -> list[tuple[float, dict]]:
        return [(t, m) for t, m in self.messages if m.get("type") == "partial"]

    @property
    def finals(self) -> list[tuple[float, dict]]:
        return [(t, m) for t, m in self.messages if m.get("type") == "final"]

    @property
    def translations(self) -> list[tuple[float, dict]]:
        return [(t, m) for t, m in self.messages
                if m.get("type") == "translation"]

    def translation_for(self, sentence_id) -> Optional[dict]:
        """The translation that belongs to a sentence, if it arrived.

        Sentences and translations are separate messages now: a sentence goes
        out as soon as it is transcribed and its translation catches up, so
        that an LLM call cannot sit on the thread reading audio.
        """
        for _t, message in self.translations:
            if message.get("sentence_id") == sentence_id:
                return message
        return None

    def translation_lags(self) -> list[tuple[int, float]]:
        """How long after its sentence each translation arrived."""
        sent_at = {m.get("sentence_id"): t for t, m in self.finals}
        return [(m.get("sentence_id"), t - sent_at[m["sentence_id"]])
                for t, m in self.translations
                if m.get("sentence_id") in sent_at]

    def lags_ms(self, gaps: Sequence[tuple[float, float]] = ()) -> list[float]:
        """How late each event arrived, relative to the audio it describes.

        ``at_ms`` counts audio the server *received*, so it advances only
        while this machine is recording. Every second the capture stalls,
        at_ms falls a second further behind the wall clock, and a naive
        subtraction hands that back as latency the server never spent.

        On a ten-minute run a single 0.5 s stall in the first second put the
        mean at 1209 ms against 391 ms for the same server minutes earlier -
        a threefold error from half a second of missing audio, and every event
        in the run carried it.

        So the stalls are subtracted. ``gaps`` is ``(seconds into the run,
        seconds waited)`` from :func:`pump_audio`; a gap shifts every event
        recorded after it and nothing before it.
        """
        return [
            (arrived - self.stream_start) * 1000.0
            - float(payload["at_ms"]) - self._stalled_before_ms(payload, gaps)
            for arrived, payload in self.vad_events
        ]

    @staticmethod
    def _stalled_before_ms(payload: dict,
                           gaps: Sequence[tuple[float, float]]) -> float:
        """Milliseconds of capture stall that precede this event's audio.

        A gap is recorded against wall-clock time, while ``at_ms`` is a
        position in the recording, so each gap has to be placed back into the
        recording: subtract the stalls that came before it.
        """
        at_ms = float(payload["at_ms"])
        stalled = 0.0
        for gap_at, waited in sorted(gaps):
            audio_position_ms = (gap_at - stalled / 1000.0) * 1000.0
            if audio_position_ms > at_ms:
                break
            stalled += waited * 1000.0
        return stalled


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
    for flag, error_key, name in (
        ("noise_filter_loaded", "noise_filter_error", "noise filter"),
        ("overlap_resolver_loaded", "", "overlap resolver"),
        ("speaker_model_loaded", "speaker_model_error", "speaker model"),
        ("language_model_loaded", "language_model_error", "language model"),
        ("asr_loaded", "asr_error", "ASR"),
        ("translation_loaded", "translation_error", "translation server"),
    ):
        report.add(
            f"Server has the {name} loaded",
            bool(payload.get(flag)),
            (payload.get(error_key) if error_key else "")
            or f"{flag}={payload.get(flag)}",
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


async def stream_for(
    url: str, device_hint: str | None, seconds: float, report: Report,
) -> tuple[StreamClient, Collected, list[tuple[float, float]]]:
    collected = Collected()
    client = StreamClient(url.rstrip("/") + "/ws/stream",
                          on_message=collected.record)
    task = asyncio.create_task(client.run())

    if not await client.wait_connected(timeout=10.0):
        report.add("Handshake accepted", False,
                   "; ".join(client.stats.errors) or "timed out")
        await client.stop()
        await task
        return client, collected, []
    report.add("Handshake accepted", True, f"session {client.session_id}")

    capture = LoopbackCapture(device_name_hint=device_hint)
    device = capture.start()
    report.add("Loopback device opened", True, device.name)

    print(f"\n  Streaming {seconds:.0f} s - PLAY SPEECH NOW ...")
    collected.stream_start = time.perf_counter()
    try:
        gaps = await pump_audio(capture, client, collected, seconds)
    finally:
        capture.stop()
        await client.stop(drain_timeout=2.0)
        await asyncio.wait_for(task, timeout=10.0)
    print()
    report_capture_gaps(capture, gaps)
    return client, collected, gaps


async def pump_audio(capture, client, collected: Collected,
                     seconds: float) -> list[tuple[float, float]]:
    """Read audio and send it, recording every stall in the reading.

    Split out from :func:`stream_for` so it can be tested. Everything around
    it needs a sound card and cannot run on the Dev PC, which is how a
    NameError on this very loop reached the Windows machine and cost a whole
    run: it is the one piece of this file no test had ever executed.

    A returned gap is ``(seconds into the run, seconds waited)``. Audio that
    would have filled a gap was never recorded at all, and that is not the
    same thing as a slow server even though the two look identical in the lag
    figures.
    """
    gaps: list[tuple[float, float]] = []
    stall = CHUNK_DURATION_MS / 1000.0 * 2
    deadline = collected.stream_start + seconds
    while time.perf_counter() < deadline:
        before = time.perf_counter()
        chunk = await asyncio.to_thread(capture.read, 0.5)
        waited = time.perf_counter() - before
        if waited > stall:
            gaps.append((before - collected.stream_start, waited))
        if chunk is None:
            continue
        client.send(chunk)
        elapsed = time.perf_counter() - collected.stream_start
        print(f"\r  t={elapsed:5.1f}s  sent={client.stats.chunks_sent:<5d} "
              f"queue={client.queued_chunks:<3d} "
              f"dropped={client.stats.chunks_dropped}  "
              f"events={len(collected.vad_events)}", end="", flush=True)
    return gaps


def report_capture_gaps(capture: LoopbackCapture,
                        gaps: list[tuple[float, float]]) -> None:
    """Where the audio went, when some of it did not arrive.

    Missing audio and a slow server look identical in the lag figures, and the
    lag figures were what the test judged. This is what tells them apart, and
    it lives on the client because that is where audio is either read or lost.
    """
    stats = capture.stats
    print(f"  Capture: {stats.callbacks} callbacks, "
          f"{stats.chunks_emitted} chunks emitted, "
          f"{stats.chunks_dropped} dropped by the capture queue, "
          f"{stats.input_overflows} device overflow(s)")
    if not gaps:
        return
    total = sum(waited for _at, waited in gaps)
    print(f"  Capture stalled {len(gaps)} time(s), {total:.1f} s in total. "
          f"The audio that would have filled those gaps was never recorded:")
    for at, waited in sorted(gaps, key=lambda gap: -gap[1])[:5]:
        print(f"    at t={at:6.1f}s  waited {waited:.2f}s for a chunk")


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


def check_events(collected: Collected, report: Report,
                 audio_seconds: float = 0.0, wall_seconds: float = 0.0,
                 gaps: Sequence[tuple[float, float]] = ()) -> None:
    """Judge the server on how fast events came back.

    ``audio_seconds`` and ``wall_seconds`` are here for a reason worth
    stating. Lag is ``arrival - (stream_start + at_ms)``, and ``at_ms`` counts
    audio the server *received*. If this machine failed to read part of the
    meeting, at_ms trails the wall clock by exactly the amount missing and
    every lag is inflated by it - a run that sent 107 s of audio over 120 s
    reported a flat 13 s lag on every event and blamed the server, which had
    done nothing wrong.

    So when audio is missing the lag figures are printed and not judged. A
    check that names the wrong culprit is worse than no check at all.
    """
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

    lags = collected.lags_ms(gaps)
    if not lags:
        return
    worst = max(lags)
    stalled = sum(waited for _at, waited in gaps)
    print(f"\n  End-to-end lag: mean {statistics.fmean(lags):.0f} ms, "
          f"max {worst:.0f} ms (budget {MAX_EVENT_LAG_MS:.0f} ms)")
    if stalled:
        uncorrected = collected.lags_ms()
        print(f"    ({stalled:.1f} s of capture stall subtracted; "
              f"uncorrected mean would read "
              f"{statistics.fmean(uncorrected):.0f} ms)")

    # Whatever the stalls do not explain. A stall can be subtracted because
    # its size is known; audio that went missing without one cannot be, and
    # then the figures cannot be scored at all.
    missing = (wall_seconds - audio_seconds - stalled) if wall_seconds else 0.0
    if missing > MISSING_AUDIO_TOLERANCE_S:
        print(f"    NOT JUDGED: {missing:.1f} s of audio never left this "
              f"machine and no capture stall accounts for it, so at_ms "
              f"trails the wall clock by an unknown amount and every lag "
              f"above is inflated by it. Fix the capture first; these "
              f"numbers say nothing about the server until then.")
        return

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


# The same word lists the server uses, read from server/data. Loaded here
# rather than copied so an entry added on one side cannot be forgotten on the
# other - but checked separately, because a test that trusts the thing it is
# testing proves nothing, and both sides once let the same line through.
_INVENTED = Hallucinations()


def is_invented(text: str) -> bool:
    return _INVENTED.is_invented(text)


#: A translation arriving further behind its sentence than this has stopped
#: belonging to it: the measured median gap between sentences is 3.58 s, so
#: five seconds is already a sentence and a half ago. The server gives up at
#: TRANSLATION_MAX_LAG_SECONDS = 10; this is the tighter number the test
#: holds it to, so drift shows up here before the server starts dropping.
MAX_TRANSLATION_LAG_S = 5.0


def japanese_ratio(text: str) -> Optional[float]:
    """Fraction of the letters that are kana or kanji, or None if no letters.

    Deliberately a copy of the server's: this test has to be able to disagree
    with the server it is testing.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    japanese = sum(
        1 for c in letters
        if unicodedata.name(c, "").startswith(("HIRAGANA", "KATAKANA", "CJK"))
    )
    return japanese / len(letters)


def in_wrong_script(translation: str, source_lang: str) -> bool:
    """A Japanese sentence must not come back in Japanese, and vice versa."""
    ratio = japanese_ratio(translation)
    if ratio is None:
        return False
    if source_lang == "ja":         # target is Vietnamese
        return ratio > 0.30
    if source_lang == "vi":         # target is Japanese
        return ratio < 0.30
    return False


def check_transcripts(collected: Collected, report: Report) -> None:
    """The whole pipeline, judged by what actually reaches the client."""
    partials, finals = collected.partials, collected.finals

    print()
    print("  Running text (partial):")
    for arrived, payload in partials[-8:]:
        print(f"    +{arrived - collected.stream_start:5.1f}s "
              f"[{payload.get('lang_code') or '?'}] "
              f"{payload.get('transcript', '')[:70]}")
    if not partials:
        print("    (none)")

    print()
    print("  Committed sentences (final):")
    for arrived, payload in finals:
        sentence_id = payload.get("sentence_id")
        print(f"    +{arrived - collected.stream_start:5.1f}s "
              f"#{sentence_id} "
              f"{payload.get('speaker_id') or '?':<16} "
              f"[{payload.get('lang_code') or '?'}]")
        # The noise filter's verdict on the audio, printed next to what
        # Whisper made of it. Sign-offs have come from utterances scored 0.03
        # while real sentences scored 0.66 and up; a few more runs decides
        # whether that separation holds.
        print(f"        said      : {payload.get('transcript', '')}"
              f"   [speech {payload.get('speech_score', 0.0):.2f}]")
        translation = collected.translation_for(sentence_id)
        if translation is None:
            # Not "refused" - never answered at all. A different failure, and
            # one that only exists now that the two are separate messages.
            print("        NO TRANSLATION MESSAGE EVER ARRIVED")
            continue
        lag = next((seconds for sid, seconds in collected.translation_lags()
                    if sid == sentence_id), None)
        if translation.get("translation", "").strip():
            print(f"        translated: {translation['translation']}"
                  + (f"   (+{lag:.1f}s)" if lag is not None else ""))
        else:
            # A blank translation with no reason beside it is a silent
            # failure, and reading one cost this project a round trip.
            print(f"        NOT translated: "
                  f"{translation.get('reason') or '(no reason given)'}")
            # And the reason alone does not say whether refusing was right.
            # "far longer than the sentence" reads the same whether the model
            # rambled or the limit was too tight for a good translation.
            if translation.get("raw"):
                print(f"        the model said: {translation['raw']}")
    if not finals:
        print("    (none)")

    report.add("The meeting produced committed sentences", bool(finals),
               f"{len(finals)} final(s)")
    if not finals:
        return

    report.add("Every committed sentence has text",
               all(m.get("transcript", "").strip() for _t, m in finals), "")
    report.add("Every committed sentence names a language",
               all(m.get("lang_code") for _t, m in finals),
               f"{sorted({m.get('lang_code') for _t, m in finals})}")
    report.add("Every committed sentence names a speaker",
               all(m.get("speaker_id") for _t, m in finals),
               f"{sorted({m.get('speaker_id') for _t, m in finals})}")

    # Sentence and translation are separate messages now, so there are three
    # outcomes rather than two: translated, refused with a reason, and no
    # answer at all. The third is new and is the one worth catching - a
    # translation that never arrives is exactly what an unbounded queue
    # falling behind would look like.
    pairs = [(m, collected.translation_for(m.get("sentence_id")))
             for _t, m in finals]
    unanswered = [final for final, translation in pairs if translation is None]
    report.add("Every sentence got an answer about its translation",
               not unanswered,
               f"{len(unanswered)} of {len(finals)} never answered: "
               f"{[m.get('transcript', '')[:24] for m in unanswered][:2]}")

    answered = [(final, translation) for final, translation in pairs
                if translation is not None]
    translated = [(f, t) for f, t in answered if t.get("translation", "").strip()]
    refused = [(f, t) for f, t in answered
               if not t.get("translation", "").strip()]
    report.add("Committed sentences come back translated",
               len(translated) == len(answered),
               f"{len(translated)}/{len(answered)} translated"
               + (f"; refused: {[t.get('reason') for _f, t in refused]}"
                  if refused else ""))
    report.add("Every untranslated sentence says why",
               all(t.get("reason") for _f, t in refused),
               f"{sum(1 for _f, t in refused if not t.get('reason'))} "
               f"silent of {len(refused)}")
    # Only a refusal that judged some text owes us the text. A model that
    # said nothing, and a sentence the model never saw, have nothing to show -
    # demanding it there is a check that cannot pass, which is what the queue
    # giving up on the last sentence turned this into.
    NEVER_ANSWERED = ("the model returned nothing",
                      "the meeting ended before this was translated",
                      "not translated in time",
                      "the translation queue was full",
                      "the translator raised")
    should_show = [
        t for _f, t in refused
        if not any(t.get("reason", "").startswith(prefix)
                   for prefix in NEVER_ANSWERED)
    ]
    report.add("Every untranslated sentence shows what the model said",
               all(t.get("raw") for t in should_show),
               f"{sum(1 for t in should_show if not t.get('raw'))} "
               f"silent of {len(should_show)}")
    same = [t for f, t in translated
            if t["translation"].strip() == f.get("transcript", "").strip()]
    report.add("A translation is not just the sentence again", not same,
               f"{len(same)} identical")

    # Checked here as well as on the server, because a test that trusts the
    # thing it is testing proves nothing. A run once returned
    # はい、今の画面の as
    # はい、現在の画面の - Japanese
    # in, Japanese out - and every check above passed it.
    wrong = [t for f, t in translated
             if in_wrong_script(t["translation"], f.get("lang_code", ""))]
    report.add("No translation came back in the language it started in",
               not wrong,
               f"{[t['translation'][:30] for t in wrong][:2]}")

    # How far the translations drift behind their sentences. This is the
    # number the split exists to keep small, and the one that would grow if
    # the queue ever started falling behind.
    # Every sentence's speech score, sorted, so the gap between what the
    # classifier heard in real speech and in an invention is visible at a
    # glance and can be argued with.
    scores = sorted(m.get("speech_score", 0.0) for _t, m in finals)
    if scores:
        print(f"\n  Speech score of each committed sentence, lowest first: "
              f"{[round(value, 2) for value in scores]}")
        print("    Whisper's inventions have come from the low end. Read the "
              "lowest few above and see whether anybody said them.")

    lags = [seconds for _sid, seconds in collected.translation_lags()]
    if lags:
        print(f"\n  Translation lag behind its sentence: "
              f"mean {statistics.fmean(lags):.2f}s, worst {max(lags):.2f}s")
        report.add("Translations keep up with their sentences",
                   max(lags) < MAX_TRANSLATION_LAG_S,
                   f"worst {max(lags):.2f}s < {MAX_TRANSLATION_LAG_S}s")

    # Checked on both, because a sign-off in the running text still reaches
    # the reader even though it never becomes a sentence.
    invented_finals = [m for _t, m in finals if is_invented(m.get("transcript", ""))]
    invented_partials = [m for _t, m in partials
                         if is_invented(m.get("transcript", ""))]
    report.add("No committed sentence is a Whisper sign-off",
               not invented_finals,
               f"{[m['transcript'][:40] for m in invented_finals][:2]}")
    report.add("No running text is a Whisper sign-off",
               not invented_partials,
               f"{[m['transcript'][:40] for m in invented_partials][:2]}")

    report.add("Running text appeared before the sentences were committed",
               bool(partials),
               f"{len(partials)} partial(s)")
    if partials and finals:
        report.add("The first partial beat the first final",
                   partials[0][0] < finals[0][0],
                   f"partial at +{partials[0][0] - collected.stream_start:.1f}s, "
                   f"final at +{finals[0][0] - collected.stream_start:.1f}s")


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

    client, collected, gaps = await stream_for(args.url, args.device,
                                               args.seconds, report)
    if client.stats.connects == 0 or not client.stats.chunks_sent:
        print("\nRESULT: FAIL - never streamed anything")
        for error in client.stats.errors:
            print(f"  - {error}")
        return 1

    check_stream(client, collected, args.seconds, report)
    check_events(collected, report,
                 audio_seconds=client.stats.chunks_sent * CHUNK_DURATION_MS / 1000.0,
                 wall_seconds=args.seconds, gaps=gaps)
    check_utterances(collected, report)
    check_transcripts(collected, report)

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
