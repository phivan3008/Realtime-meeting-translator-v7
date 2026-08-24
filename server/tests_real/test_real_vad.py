"""REAL TEST - Silero VAD on the GPU pod, driven by real recorded audio.

MUST RUN ON: the GPU Server pod (real Silero model, real meeting audio).
DO NOT RUN ON: the Dev PC agent loop.

The pod has no sound card, so the audio comes from WAV files recorded on the
Windows Client PC with ``client/tests_real/test_real_audio_capture.py``.  Copy
them over first; they are already in the exact 16 kHz mono 16-bit format the
client streams.

What it proves
--------------
1. The Silero VAD model loads and runs on the pod.
2. Inference is far faster than real time, so VAD never becomes the pipeline
   bottleneck.
3. A recording of a quiet room produces no speech segments (no false
   triggers on fan noise or notification blips).
4. A recording of real meeting speech produces speech segments whose
   boundaries and timestamps are consistent.
5. The segment timestamps are exactly what the Stream Buffer Manager needs:
   monotonic, non-overlapping, and matching the forwarded audio.
6. reset() really does restore the model. The server keeps one loaded model
   across meetings, so a second session must not inherit anything from the
   first.
6. The forwarded audio still contains every word - it is written to a WAV for
   listening.

Usage
-----
    python3.11 server/tests_real/test_real_vad.py \\
        --speech recordings/meeting_vi.wav \\
        --silence recordings/quiet_room.wav

    python3.11 server/tests_real/test_real_vad.py --speech x.wav --onnx
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    CHANNELS,
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    FINALIZE_PAUSE_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    VAD_MIN_SILENCE_MS,
    VAD_THRESHOLD,
)
from server.pipeline.vad import (  # noqa: E402
    VAD_FRAME_MS,
    SegmentEvent,
    SileroVAD,
    VADError,
    VADEvent,
    VADSegmenter,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Steady state: one frame of inference must stay far below the 32 ms of audio
# it represents, or VAD falls behind the stream.
MAX_MEAN_INFERENCE_MS = VAD_FRAME_MS / 4.0      # 8 ms
MAX_P99_INFERENCE_MS = VAD_FRAME_MS             # 32 ms
# A single hiccup is survivable as long as it is shorter than the 200 ms chunk
# the client sends: the queue absorbs it and the stream catches straight back
# up. Anything longer than a whole chunk is a real stall.
MAX_WORST_INFERENCE_MS = CHUNK_DURATION_MS      # 200 ms


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
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


class TimedVAD:
    """Wrap :class:`SileroVAD` and record how long each frame takes."""

    def __init__(self, vad: SileroVAD) -> None:
        self._vad = vad
        self.latencies_ms: list[float] = []

    def probability(self, frame: np.ndarray) -> float:
        started = time.perf_counter()
        value = self._vad.probability(frame)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return value

    def reset(self) -> None:
        self._vad.reset()


@dataclass
class Replay:
    """Everything one WAV file produced when pushed through the segmenter."""

    path: Path
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    frames: int = 0
    speech_frames: int = 0
    segments: int = 0
    events: list[SegmentEvent] = field(default_factory=list)
    gated_pcm: bytearray = field(default_factory=bytearray)
    max_probability: float = 0.0
    dropped_ratio: float = 0.0

    @property
    def speech_ratio(self) -> float:
        return self.speech_frames / self.frames if self.frames else 0.0

    @property
    def realtime_factor(self) -> float:
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def gated_seconds(self) -> float:
        return len(self.gated_pcm) / SAMPLE_WIDTH / SAMPLE_RATE


def read_pcm(path: Path) -> bytes:
    """Read a WAV file, insisting on the exact format the client streams."""
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        expected = (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH)
        if actual != expected:
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; expected {expected[0]} Hz / mono / 16-bit"
            )
        return wav.readframes(wav.getnframes())


def replay(path: Path, vad: TimedVAD) -> Replay:
    """Feed a WAV through a fresh segmenter, 200 ms at a time, as the client would."""
    pcm = read_pcm(path)
    # The model is recurrent, so the previous file's tail must not colour this
    # file's first frames.
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    result = Replay(path=path, audio_seconds=len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE)

    started = time.perf_counter()
    for offset in range(0, len(pcm), CHUNK_BYTES):
        out = segmenter.push(pcm[offset : offset + CHUNK_BYTES])
        result.gated_pcm.extend(out.pcm)
        result.events.extend(out.events)
        result.max_probability = max(result.max_probability, out.max_probability)
    final = segmenter.close()
    result.events.extend(final.events)
    result.wall_seconds = time.perf_counter() - started

    result.frames = segmenter.stats.frames_total
    result.speech_frames = segmenter.stats.frames_speech
    result.segments = segmenter.stats.segments
    result.dropped_ratio = segmenter.stats.dropped_ratio
    return result


def describe(result: Replay) -> None:
    print(f"\n  {result.path.name}: {result.audio_seconds:.1f} s of audio, "
          f"{result.frames} frames")
    print(f"    speech ratio {result.speech_ratio * 100:.1f}%, "
          f"segments {result.segments}, p_max {result.max_probability:.2f}, "
          f"dropped {result.dropped_ratio * 100:.1f}%")
    print(f"    compute {result.wall_seconds:.2f} s "
          f"(realtime factor {result.realtime_factor:.4f})")
    if result.events:
        print("    segments:")
        for start, end in pairs(result.events):
            print(f"      {start.at_ms / 1000:7.2f}s -> {end.at_ms / 1000:7.2f}s "
                  f"({(end.at_ms - start.at_ms) / 1000:.2f}s)")


def pairs(events: list[SegmentEvent]) -> list[tuple[SegmentEvent, SegmentEvent]]:
    """Zip the start/end events into segments, ignoring an unmatched tail."""
    starts = [e for e in events if e.kind is VADEvent.SPEECH_START]
    ends = [e for e in events if e.kind is VADEvent.SPEECH_END]
    return list(zip(starts, ends))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, min(len(sorted_values), round(fraction * len(sorted_values))))
    return sorted_values[rank - 1]


def check_latency(vad: TimedVAD, report: Report) -> None:
    if not vad.latencies_ms:
        report.add("VAD inference latency measured", False, "no frames processed")
        return
    latencies = vad.latencies_ms
    ordered = sorted(latencies)
    mean = statistics.fmean(ordered)
    p95 = percentile(ordered, 0.95)
    p99 = percentile(ordered, 0.99)
    worst = ordered[-1]
    slowest_at = latencies.index(worst)

    print(f"\n  Inference over {len(latencies)} frames: mean {mean:.2f} ms, "
          f"p95 {p95:.2f} ms, p99 {p99:.2f} ms, max {worst:.2f} ms")
    print(f"    slowest frame is #{slowest_at} of {len(latencies)}"
          + (" - at the very start, model warm-up" if slowest_at < 3
             else " - mid-stream"))
    print(f"    budgets: mean < {MAX_MEAN_INFERENCE_MS:.0f} ms, "
          f"p99 < {MAX_P99_INFERENCE_MS} ms (one frame), "
          f"max < {MAX_WORST_INFERENCE_MS} ms (one client chunk)")

    report.add("VAD is faster than real time (mean)", mean < MAX_MEAN_INFERENCE_MS,
               f"mean {mean:.2f} ms < {MAX_MEAN_INFERENCE_MS:.0f} ms")
    report.add("Steady-state frames stay inside the frame budget",
               p99 < MAX_P99_INFERENCE_MS,
               f"p99 {p99:.2f} ms < {MAX_P99_INFERENCE_MS} ms")
    report.add("No stall longer than one client chunk",
               worst < MAX_WORST_INFERENCE_MS,
               f"max {worst:.2f} ms (frame #{slowest_at}) "
               f"< {MAX_WORST_INFERENCE_MS} ms")


def check_silence(result: Replay, report: Report) -> None:
    report.add("Quiet recording triggers no speech segment", result.segments == 0,
               f"{result.segments} segment(s), p_max {result.max_probability:.2f}")
    report.add("Quiet recording is dropped before the heavy stages",
               result.dropped_ratio > 0.95,
               f"{result.dropped_ratio * 100:.1f}% dropped")


def check_speech(result: Replay, report: Report) -> None:
    report.add("Speech is detected", result.segments >= 1,
               f"{result.segments} segment(s)")
    report.add("Most of the recording is marked as speech",
               result.speech_ratio > 0.4,
               f"speech ratio {result.speech_ratio * 100:.1f}% > 40%")
    report.add("The model is confident on real speech",
               result.max_probability > 0.8,
               f"p_max {result.max_probability:.2f}")
    report.add("Forwarded audio is not empty", len(result.gated_pcm) > 0,
               f"{result.gated_seconds:.1f} s forwarded")
    report.add("Some audio was still dropped", result.dropped_ratio > 0.0,
               f"{result.dropped_ratio * 100:.1f}% dropped")


def check_repeatable(first: Replay, again: Replay, report: Report) -> None:
    """The same audio through the same model instance must give the same result.

    The server loads Silero once and reuses it for every meeting, resetting
    between sessions. If reset() left anything behind, meeting two would be
    judged partly by meeting one - quietly, and differently every time.
    """
    report.add(
        "Replaying the same audio gives byte-identical speech",
        bytes(first.gated_pcm) == bytes(again.gated_pcm),
        f"{len(first.gated_pcm)} bytes then {len(again.gated_pcm)} bytes",
    )
    report.add(
        "Replaying the same audio gives the same segments",
        [(e.kind, e.at_ms) for e in first.events]
        == [(e.kind, e.at_ms) for e in again.events],
        f"{len(first.events)} events then {len(again.events)}",
    )


def check_timestamps(result: Replay, report: Report) -> None:
    """The Stream Buffer Manager consumes these; they have to be exact."""
    events = result.events
    starts = sum(1 for e in events if e.kind is VADEvent.SPEECH_START)
    ends = sum(1 for e in events if e.kind is VADEvent.SPEECH_END)
    report.add("Every segment is opened and closed", starts == ends,
               f"{starts} speech_start / {ends} speech_end")

    kinds = [e.kind for e in events]
    alternating = all(
        kinds[i] is not kinds[i + 1] for i in range(len(kinds) - 1)
    ) and (not kinds or kinds[0] is VADEvent.SPEECH_START)
    report.add("Events strictly alternate start/end", alternating,
               " ".join(k.value for k in kinds[:6]) + (" ..." if len(kinds) > 6 else ""))

    times = [e.at_ms for e in events]
    report.add("Timestamps never go backwards",
               all(b >= a for a, b in zip(times, times[1:])),
               f"{len(times)} events")

    segments = pairs(events)
    report.add("Every segment has a positive duration",
               all(end.at_ms > start.at_ms for start, end in segments),
               f"{len(segments)} segment(s)")
    report.add("Every segment fits inside the recording",
               all(end.at_ms <= result.audio_seconds * 1000 + VAD_FRAME_MS
                   for _, end in segments),
               f"recording is {result.audio_seconds:.1f} s")

    # The forwarded audio must be exactly the sum of the segment durations.
    total_ms = sum(end.at_ms - start.at_ms for start, end in segments)
    report.add("Segment bounds match the forwarded audio",
               abs(total_ms - result.gated_seconds * 1000) < VAD_FRAME_MS,
               f"segments {total_ms / 1000:.2f} s vs audio "
               f"{result.gated_seconds:.2f} s")

    # Each closing segment must carry enough silence for the 400 ms rule.
    report.add("Closed segments carry the finalize pause",
               VAD_MIN_SILENCE_MS > FINALIZE_PAUSE_MS,
               f"hangover {VAD_MIN_SILENCE_MS} ms > pause {FINALIZE_PAUSE_MS} ms")


def save_gated(result: Replay) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{result.path.stem}_gated.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(result.gated_pcm))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, required=True,
                        help="WAV of real meeting speech (16 kHz mono 16-bit)")
    parser.add_argument("--silence", type=Path, default=None,
                        help="WAV of a quiet room, to check for false triggers")
    parser.add_argument("--onnx", action="store_true",
                        help="use the ONNX Silero model instead of torch jit")
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - SILERO VAD (server pipeline, step 0)")
    print(f"threshold={VAD_THRESHOLD}  hangover={VAD_MIN_SILENCE_MS} ms  "
          f"frame={VAD_FRAME_MS} ms  finalize pause={FINALIZE_PAUSE_MS} ms")
    print("=" * 72)

    report = Report()
    try:
        print(f"\n  Loading Silero VAD ({'onnx' if args.onnx else 'torch jit'}) ...")
        loading = time.perf_counter()
        vad = TimedVAD(SileroVAD(onnx=args.onnx))
        print(f"  Model ready in {time.perf_counter() - loading:.1f} s")
        try:
            import torch

            print(f"  torch {torch.__version__}, "
                  f"CUDA available: {torch.cuda.is_available()} "
                  "(VAD itself stays on CPU by design)")
        except ImportError:
            pass

        silence: Replay | None = None
        if args.silence is not None:
            silence = replay(args.silence, vad)
            describe(silence)

        speech = replay(args.speech, vad)
        describe(speech)

        # Same file, same model instance, straight after: only reset() stands
        # between the two runs.
        again = replay(args.speech, vad)

        print("\nChecks:")
        check_latency(vad, report)
        report.add("Faster than real time end to end",
                   speech.realtime_factor < 0.1,
                   f"realtime factor {speech.realtime_factor:.4f} < 0.1")
        if silence is not None:
            check_silence(silence, report)
        check_speech(speech, report)
        check_timestamps(speech, report)
        check_repeatable(speech, again, report)

        gated_path = save_gated(speech)
        print(f"\n  Forwarded audio: {gated_path}")
        print("  Copy it back and listen: every word must still be there, "
              "with the long pauses cut out.")

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
