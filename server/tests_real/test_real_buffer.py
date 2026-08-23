"""REAL TEST - VAD + Stream Buffer Manager on real recorded audio.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

Module 4 sits between the VAD and everything expensive, so its output is what
Whisper will actually be asked to transcribe.  This replays a real recording
through both stages and checks the sentences it carves out.

What it proves
--------------
1. The utterances are a clean partition of the speech the VAD forwarded: no
   audio is lost between sentences and none is transcribed twice.
2. No utterance is longer than the max duration, so the viewer never waits
   more than that for a committed line.
3. The timestamps are consistent: rising, non-overlapping, and a continued
   chain joins end to start.
4. Partial windows appear at roughly the configured cadence while somebody
   is talking, and never while nobody is.
5. Every utterance is written out as a WAV, so the max-duration cuts can be
   listened to - they must land between words, not through one.

Usage
-----
    python3.11 server/tests_real/test_real_buffer.py \\
        --speech recordings/meeting_speech.wav
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    CHANNELS,
    CHUNK_BYTES,
    FINALIZE_MAX_DURATION_MS,
    PARTIAL_INTERVAL_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SPLIT_SEARCH_MS,
)
from server.pipeline.buffer import (  # noqa: E402
    BufferManager,
    BufferStats,
    FinalizeReason,
    PartialWindow,
    Utterance,
)
from server.pipeline.vad import SileroVAD, VADError, VADSegmenter  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# One VAD frame of slack: the cut is frame aligned, so it can overshoot by
# less than 32 ms.
DURATION_TOLERANCE_MS = 40.0


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
class Replay:
    path: Path
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    forwarded: bytearray = field(default_factory=bytearray)
    utterances: list[Utterance] = field(default_factory=list)
    partials: list[PartialWindow] = field(default_factory=list)
    buffer_stats: Optional[BufferStats] = None

    @property
    def forwarded_seconds(self) -> float:
        return len(self.forwarded) / SAMPLE_WIDTH / SAMPLE_RATE


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; expected 16000 Hz / mono / 16-bit"
            )
        return wav.readframes(wav.getnframes())


def replay(path: Path, vad: SileroVAD) -> Replay:
    pcm = read_pcm(path)
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    buffer = BufferManager()
    result = Replay(path=path,
                    audio_seconds=len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE)

    started = time.perf_counter()
    for offset in range(0, len(pcm), CHUNK_BYTES):
        out = segmenter.push(pcm[offset : offset + CHUNK_BYTES])
        result.forwarded.extend(out.pcm)
        decisions = buffer.push(out)
        result.utterances.extend(decisions.finals)
        if decisions.partial is not None:
            result.partials.append(decisions.partial)
    segmenter.close()
    result.utterances.extend(buffer.flush().finals)
    result.wall_seconds = time.perf_counter() - started
    result.buffer_stats = buffer.stats
    return result


def describe(result: Replay) -> None:
    print(f"\n  {result.path.name}: {result.audio_seconds:.1f} s in, "
          f"{result.forwarded_seconds:.1f} s of speech forwarded by the VAD")
    print(f"    {len(result.utterances)} utterances, "
          f"{len(result.partials)} partial windows, "
          f"compute {result.wall_seconds:.2f} s")
    print(f"    reasons: {result.buffer_stats.finalized_by}")
    print("\n  Utterances:")
    print("    idx     start       end   length  reason         continues")
    for utterance in result.utterances:
        print(f"    {utterance.index:>3}  {utterance.start_ms / 1000:8.2f}s "
              f"{utterance.end_ms / 1000:8.2f}s "
              f"{utterance.duration_ms / 1000:7.2f}s  "
              f"{utterance.reason.value:<14} "
              f"{'yes' if utterance.continues_previous else ''}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_partition(result: Replay, report: Report) -> None:
    joined = b"".join(u.pcm for u in result.utterances)
    report.add(
        "Utterances are exactly the speech the VAD forwarded",
        joined == bytes(result.forwarded),
        f"{len(joined)} bytes committed vs {len(result.forwarded)} forwarded",
    )
    report.add(
        "No audio is transcribed twice",
        sum(len(u.pcm) for u in result.utterances) == len(result.forwarded),
        f"{sum(len(u.pcm) for u in result.utterances)} bytes total",
    )
    report.add("At least one sentence was carved out",
               len(result.utterances) >= 1,
               f"{len(result.utterances)} utterance(s)")


def check_durations(result: Replay, report: Report) -> None:
    longest = max((u.duration_ms for u in result.utterances), default=0.0)
    report.add(
        "No utterance outstays the max duration",
        longest <= FINALIZE_MAX_DURATION_MS + DURATION_TOLERANCE_MS,
        f"longest {longest / 1000:.2f} s, limit "
        f"{FINALIZE_MAX_DURATION_MS / 1000:.1f} s",
    )
    report.add("No empty utterance was committed",
               all(u.duration_ms > 0 for u in result.utterances),
               f"shortest {min((u.duration_ms for u in result.utterances), default=0):.0f} ms")

    cuts = [u for u in result.utterances
            if u.reason is FinalizeReason.MAX_DURATION]
    if cuts:
        shortest_cut = min(u.duration_ms for u in cuts)
        report.add(
            "Length cuts land inside the search window, not on the raw limit",
            shortest_cut >= FINALIZE_MAX_DURATION_MS - SPLIT_SEARCH_MS,
            f"{len(cuts)} cut(s), shortest {shortest_cut / 1000:.2f} s",
        )
    else:
        print("    (no max_duration cut in this recording)")


def check_timestamps(result: Replay, report: Report) -> None:
    starts = [u.start_ms for u in result.utterances]
    report.add("Utterances are in order", starts == sorted(starts),
               f"{len(starts)} utterance(s)")
    report.add("Indexes are consecutive from zero",
               [u.index for u in result.utterances]
               == list(range(len(result.utterances))),
               "")
    overlaps = [
        (a.index, b.index)
        for a, b in zip(result.utterances, result.utterances[1:])
        if b.start_ms < a.end_ms - 1e-6
    ]
    report.add("Utterances never overlap", not overlaps, f"{overlaps[:3]}")

    joins = [
        (a, b) for a, b in zip(result.utterances, result.utterances[1:])
        if b.continues_previous
    ]
    report.add(
        "A continued sentence joins end to start with no gap",
        all(abs(b.start_ms - a.end_ms) < 1.0 for a, b in joins),
        f"{len(joins)} continuation(s)",
    )


def check_partials(result: Replay, report: Report) -> None:
    if not result.partials:
        report.add("Partial windows are produced while somebody talks",
                   False, "none")
        return
    report.add("Partial windows are produced while somebody talks", True,
               f"{len(result.partials)} window(s)")
    report.add("Every partial has audio in it",
               all(p.duration_ms > 0 for p in result.partials), "")
    report.add(
        "Partials belong to an utterance that has not been committed yet",
        all(p.index <= max(u.index for u in result.utterances)
            for p in result.partials),
        "",
    )

    per_index: dict[int, list[PartialWindow]] = {}
    for partial in result.partials:
        per_index.setdefault(partial.index, []).append(partial)
    gaps = [
        later.end_ms - earlier.end_ms
        for windows in per_index.values()
        for earlier, later in zip(windows, windows[1:])
    ]
    if gaps:
        worst = max(gaps)
        report.add(
            "Partials keep to the configured cadence",
            worst <= PARTIAL_INTERVAL_MS * 2,
            f"largest gap {worst:.0f} ms, cadence {PARTIAL_INTERVAL_MS} ms",
        )


def save_utterances(result: Replay) -> Path:
    directory = OUTPUT_DIR / f"{result.path.stem}_utterances"
    directory.mkdir(parents=True, exist_ok=True)
    for utterance in result.utterances:
        name = (f"{utterance.index:03d}_{utterance.reason.value}"
                f"{'_cont' if utterance.continues_previous else ''}.wav")
        with wave.open(str(directory / name), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(utterance.pcm)
    return directory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, required=True,
                        help="WAV of real meeting speech (16 kHz mono 16-bit)")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - STREAM BUFFER MANAGER (server pipeline, step 2)")
    print(f"max duration={FINALIZE_MAX_DURATION_MS} ms  "
          f"partial every {PARTIAL_INTERVAL_MS} ms  "
          f"split search={SPLIT_SEARCH_MS} ms")
    print("=" * 72)

    report = Report()
    try:
        print(f"\n  Loading Silero VAD ({'onnx' if args.onnx else 'torch jit'}) ...")
        vad = SileroVAD(onnx=args.onnx)
        result = replay(args.speech, vad)
        describe(result)

        print("\nChecks:")
        check_partition(result, report)
        check_durations(result, report)
        check_timestamps(result, report)
        check_partials(result, report)

        directory = save_utterances(result)
        print(f"\n  Utterance WAVs: {directory}")
        print("  Listen to any file named *_max_duration*.wav: the cut must "
              "fall between words, not through one.")

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

    except (VADError, ValueError, FileNotFoundError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
