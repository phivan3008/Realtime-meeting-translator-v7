"""REAL HARDWARE TEST - Silero VAD on live loopback audio.

MUST RUN ON: the Windows Client PC (real sound card + real Silero model).
DO NOT RUN ON: the Dev PC agent loop or the GPU server.

What it proves
--------------
1. The Silero VAD model loads and runs on the client CPU.
2. Inference is far faster than real time (one 32 ms frame must cost well
   under 32 ms, otherwise the client cannot keep up with the capture).
3. A quiet room produces no speech segments (no false triggers).
4. Real meeting speech produces speech segments with sensible boundaries.
5. Silence is actually removed from the outgoing stream, and the removed
   pauses are announced as ``speech_end`` events so the server can still
   finalise sentences.
6. The gated audio still contains every word: it is written to a WAV file for
   listening.

Usage
-----
    py -3.11 client\\tests_real\\test_real_vad.py
    py -3.11 client\\tests_real\\test_real_vad.py --onnx
    py -3.11 client\\tests_real\\test_real_vad.py --wav path\\to\\loopback.wav

The interactive run has two phases and tells you what to do in each one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.audio.capture import (  # noqa: E402
    AudioCaptureError,
    LoopbackCapture,
    pcm16_to_wav_bytes,
)
from client.audio.vad import (  # noqa: E402
    VAD_FRAME_MS,
    SileroVAD,
    VADError,
    VADEvent,
    VADGate,
)
from client.config import (  # noqa: E402
    CHUNK_BYTES,
    TARGET_SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_THRESHOLD,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SILENCE_SECONDS = 6.0
SPEECH_SECONDS = 15.0

# One frame of inference must stay far below the 32 ms it represents.
MAX_MEAN_INFERENCE_MS = VAD_FRAME_MS / 4.0      # 8 ms
MAX_WORST_INFERENCE_MS = VAD_FRAME_MS           # 32 ms


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
class PhaseResult:
    label: str
    seconds: float = 0.0
    frames: int = 0
    speech_frames: int = 0
    segments: int = 0
    events: list[tuple[float, VADEvent]] = field(default_factory=list)
    raw_pcm: bytearray = field(default_factory=bytearray)
    gated_pcm: bytearray = field(default_factory=bytearray)
    max_probability: float = 0.0

    @property
    def speech_ratio(self) -> float:
        return self.speech_frames / self.frames if self.frames else 0.0

    @property
    def bytes_saved(self) -> float:
        if not self.raw_pcm:
            return 0.0
        return 1.0 - len(self.gated_pcm) / len(self.raw_pcm)


def run_phase(capture: LoopbackCapture, gate: VADGate, seconds: float,
              label: str, instruction: str) -> PhaseResult:
    """Capture for ``seconds`` while pushing everything through the gate."""
    print(f"\n  --- {label} ({seconds:.0f}s) ---")
    print(f"  {instruction}")
    for count in (3, 2, 1):
        print(f"\r  starting in {count} ...", end="", flush=True)
        time.sleep(1)
    print("\r  GO" + " " * 20)

    result = PhaseResult(label=label)
    before = (gate.stats.frames_total, gate.stats.frames_speech, gate.stats.segments)
    started = time.perf_counter()

    while time.perf_counter() - started < seconds:
        chunk = capture.read(timeout=1.0)
        if chunk is None:
            continue
        elapsed = time.perf_counter() - started
        out = gate.push(chunk)
        result.raw_pcm.extend(chunk)
        result.gated_pcm.extend(out.pcm)
        result.max_probability = max(result.max_probability, out.max_probability)
        for event in out.events:
            result.events.append((elapsed, event))
            print(f"\r  t={elapsed:5.1f}s  {event.value:<12}"
                  f"  p_max={out.max_probability:.2f}")
        print(f"\r  t={elapsed:5.1f}s  speech={out.is_speech!s:<5} "
              f"p_max={out.max_probability:.2f}  sent={len(out.pcm):>5}B",
              end="", flush=True)

    result.seconds = time.perf_counter() - started
    result.frames = gate.stats.frames_total - before[0]
    result.speech_frames = gate.stats.frames_speech - before[1]
    result.segments = gate.stats.segments - before[2]
    print()
    return result


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_latency(timed: TimedVAD, report: Report) -> None:
    if not timed.latencies_ms:
        report.add("VAD inference latency measured", False, "no frames processed")
        return
    latencies = sorted(timed.latencies_ms)
    mean = statistics.fmean(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    worst = latencies[-1]
    print(f"\n  Inference over {len(latencies)} frames: mean {mean:.2f} ms, "
          f"p95 {p95:.2f} ms, max {worst:.2f} ms "
          f"(budget {VAD_FRAME_MS} ms per frame)")
    report.add("VAD is faster than real time (mean)", mean < MAX_MEAN_INFERENCE_MS,
               f"mean {mean:.2f} ms < {MAX_MEAN_INFERENCE_MS:.0f} ms")
    report.add("No frame blows the real-time budget",
               worst < MAX_WORST_INFERENCE_MS,
               f"max {worst:.2f} ms < {MAX_WORST_INFERENCE_MS:.0f} ms")


def check_silence_phase(phase: PhaseResult, report: Report) -> None:
    print(f"\n  {phase.label}: {phase.frames} frames, "
          f"speech ratio {phase.speech_ratio * 100:.1f}%, "
          f"segments {phase.segments}, p_max {phase.max_probability:.2f}")
    report.add("Quiet room triggers no speech segment", phase.segments == 0,
               f"{phase.segments} segment(s), p_max {phase.max_probability:.2f}")
    report.add("Quiet room sends (almost) nothing", phase.bytes_saved > 0.95,
               f"{phase.bytes_saved * 100:.1f}% of the bytes dropped")


def check_speech_phase(phase: PhaseResult, report: Report) -> None:
    print(f"\n  {phase.label}: {phase.frames} frames, "
          f"speech ratio {phase.speech_ratio * 100:.1f}%, "
          f"segments {phase.segments}, p_max {phase.max_probability:.2f}")
    report.add("Speech is detected", phase.segments >= 1,
               f"{phase.segments} segment(s)")
    report.add("Most of the speech phase is marked as speech",
               phase.speech_ratio > 0.4,
               f"speech ratio {phase.speech_ratio * 100:.1f}% > 40%")
    report.add("The model is confident on real speech",
               phase.max_probability > 0.8,
               f"p_max {phase.max_probability:.2f}")
    report.add("Gated audio is not empty", len(phase.gated_pcm) > 0,
               f"{len(phase.gated_pcm)} bytes forwarded")


def check_events(phases: list[PhaseResult], report: Report) -> None:
    events = [e for phase in phases for _, e in phase.events]
    starts = events.count(VADEvent.SPEECH_START)
    ends = events.count(VADEvent.SPEECH_END)
    report.add("Every speech_start is announced", starts >= 1,
               f"{starts} speech_start, {ends} speech_end")
    report.add("Events alternate correctly", abs(starts - ends) <= 1,
               f"{starts} start / {ends} end")


def save_wavs(phase: PhaseResult, stamp: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / f"vad_{stamp}_raw.wav"
    gated_path = OUTPUT_DIR / f"vad_{stamp}_gated.wav"
    raw_path.write_bytes(pcm16_to_wav_bytes(bytes(phase.raw_pcm)))
    gated_path.write_bytes(pcm16_to_wav_bytes(bytes(phase.gated_pcm)))
    return raw_path, gated_path


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------
def run_on_wav(path: Path, gate: VADGate, timed: TimedVAD, report: Report) -> int:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            TARGET_SAMPLE_RATE, 1, 2
        ):
            print(f"  {path} is not 16 kHz mono 16-bit PCM")
            return 2
        pcm = wav.readframes(wav.getnframes())

    print(f"\n  Replaying {len(pcm) / 2 / TARGET_SAMPLE_RATE:.1f} s from {path.name}")
    phase = PhaseResult(label="WAV replay")
    started = time.perf_counter()
    for offset in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[offset : offset + CHUNK_BYTES]
        out = gate.push(chunk)
        phase.raw_pcm.extend(chunk)
        phase.gated_pcm.extend(out.pcm)
        phase.max_probability = max(phase.max_probability, out.max_probability)
        for event in out.events:
            phase.events.append(
                (offset / 2 / TARGET_SAMPLE_RATE, event)
            )
    wall = time.perf_counter() - started
    phase.frames = gate.stats.frames_total
    phase.speech_frames = gate.stats.frames_speech
    phase.segments = gate.stats.segments
    audio_seconds = len(pcm) / 2 / TARGET_SAMPLE_RATE

    print("\n  Events:")
    for at, event in phase.events:
        print(f"    t={at:6.2f}s  {event.value}")

    print("\nChecks:")
    check_latency(timed, report)
    report.add("Faster than real time end to end", wall < audio_seconds,
               f"{wall:.2f} s of compute for {audio_seconds:.2f} s of audio "
               f"(RTF {wall / audio_seconds:.3f})")
    check_speech_phase(phase, report)
    check_events([phase], report)

    raw_path, gated_path = save_wavs(
        phase, f"replay_{datetime.now():%Y%m%d_%H%M%S}"
    )
    print(f"\n  Gated audio: {gated_path}")
    return 1 if report.failed else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", action="store_true",
                        help="use the ONNX Silero model instead of torch jit")
    parser.add_argument("--wav", type=Path, default=None,
                        help="replay a 16 kHz mono WAV instead of capturing live")
    parser.add_argument("--device", default=None,
                        help="substring of the loopback device name to use")
    parser.add_argument("--silence-seconds", type=float, default=SILENCE_SECONDS)
    parser.add_argument("--speech-seconds", type=float, default=SPEECH_SECONDS)
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - SILERO VAD")
    print(f"threshold={VAD_THRESHOLD}  hangover={VAD_MIN_SILENCE_MS} ms  "
          f"frame={VAD_FRAME_MS} ms")
    print("=" * 72)

    report = Report()
    try:
        print(f"\n  Loading Silero VAD ({'onnx' if args.onnx else 'torch jit'}) ...")
        loading = time.perf_counter()
        timed = TimedVAD(SileroVAD(onnx=args.onnx))
        print(f"  Model ready in {time.perf_counter() - loading:.1f} s")
        gate = VADGate(vad=timed)

        if args.wav is not None:
            return run_on_wav(args.wav, gate, timed, report)

        capture = LoopbackCapture(device_name_hint=args.device)
        device = capture.start()
        print(f"\n  Capturing from: {device.name}")
        try:
            silence = run_phase(
                capture, gate, args.silence_seconds, "PHASE 1 - SILENCE",
                "Stop all audio. Do not speak, do not play anything.",
            )
            speech = run_phase(
                capture, gate, args.speech_seconds, "PHASE 2 - SPEECH",
                "Play a meeting recording / talk with pauses between sentences.",
            )
        finally:
            capture.stop()
        gate.close()

        print("\n  Events (phase 2):")
        for at, event in speech.events:
            print(f"    t={at:6.2f}s  {event.value}")

        print("\nChecks:")
        check_latency(timed, report)
        check_silence_phase(silence, report)
        check_speech_phase(speech, report)
        check_events([silence, speech], report)
        report.add(
            "Gating saves bandwidth over the whole session",
            gate.stats.bandwidth_saved > 0.1,
            f"{gate.stats.bandwidth_saved * 100:.1f}% of the captured bytes "
            "never left the client",
        )

        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        raw_path, gated_path = save_wavs(speech, stamp)
        print(f"\n  Raw phase-2 audio  : {raw_path}")
        print(f"  Gated phase-2 audio: {gated_path}")
        print("  Listen to the gated file: every word must still be there, "
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

    except (AudioCaptureError, VADError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
