"""REAL HARDWARE TEST - WASAPI loopback capture.

MUST RUN ON: the Windows Client PC (a real sound card is required).
DO NOT RUN ON: the Dev PC agent loop or the GPU server.

What it proves
--------------
1. A WASAPI loopback endpoint for the default playback device can be opened.
2. Audio is delivered continuously for the whole recording window.
3. Every chunk handed to the network layer is exactly 200 ms of 16 kHz mono
   16-bit PCM (6400 bytes).
4. The stream keeps up with real time (no starvation, no growing backlog,
   no dropped chunks).
5. The captured signal actually contains sound, not digital silence.

Usage
-----
    py -3.11 client/tests_real/test_real_audio_capture.py
    py -3.11 client/tests_real/test_real_audio_capture.py --list
    py -3.11 client/tests_real/test_real_audio_capture.py --seconds 15 --device "Realtek"

The recording is written to ``client/tests_real/output/`` so it can be played
back and listened to afterwards.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.audio.capture import (  # noqa: E402
    AudioCaptureError,
    LoopbackCapture,
    find_loopback_device,
    list_loopback_devices,
    pcm16_to_wav_bytes,
)
from client.config import (  # noqa: E402
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    TARGET_SAMPLE_RATE,
)

DEFAULT_SECONDS = 10.0
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Signal level thresholds, in dBFS.
MIN_PEAK_DBFS = -50.0        # below this the meeting audio is effectively muted
MIN_RMS_DBFS = -60.0
# Tolerated difference between wall clock time and captured audio duration.
MAX_TIMING_DRIFT_RATIO = 0.05


# ---------------------------------------------------------------------------
# Tiny check harness
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
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
        return passed

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def dbfs(value: float) -> float:
    """Convert a linear 0..1 amplitude to dBFS (-inf guarded)."""
    return 20.0 * np.log10(value) if value > 0 else -999.0


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def print_devices() -> None:
    devices = list_loopback_devices()
    print(f"\nWASAPI loopback devices ({len(devices)}):")
    for dev in devices:
        print(
            f"  [{dev.index:>3}] {dev.name}  "
            f"({dev.channels} ch @ {dev.default_sample_rate} Hz)"
        )
    try:
        print(f"\nDefault choice: {find_loopback_device().name}")
    except AudioCaptureError as exc:
        print(f"\nDefault choice unavailable: {exc}")


def run_capture(seconds: float, device_hint: str | None, report: Report) -> bytes:
    """Record for ``seconds`` and return the concatenated 16 kHz mono PCM."""
    capture = LoopbackCapture(device_name_hint=device_hint)
    device = capture.start()

    report.add(
        "Loopback device opened",
        True,
        f"{device.name} ({device.channels} ch @ {device.default_sample_rate} Hz, "
        f"{capture.stats.device_format})",
    )

    print(f"\n  Recording {seconds:.0f} s - PLAY AUDIO NOW ...")
    started = time.perf_counter()
    deadline = started + seconds
    pcm = bytearray()
    bad_sizes = 0
    max_queue_depth = 0
    max_gap_ms = 0.0
    last_arrival = started

    while time.perf_counter() < deadline:
        chunk = capture.read(timeout=1.0)
        if chunk is None:
            continue
        now = time.perf_counter()
        max_gap_ms = max(max_gap_ms, (now - last_arrival) * 1000.0)
        last_arrival = now
        max_queue_depth = max(max_queue_depth, capture.queued_chunks)
        if len(chunk) != CHUNK_BYTES:
            bad_sizes += 1
        pcm.extend(chunk)
        elapsed = now - started
        print(
            f"\r  t={elapsed:5.1f}s  chunks={capture.stats.chunks_emitted:<5d} "
            f"queue={capture.queued_chunks:<3d} dropped={capture.stats.chunks_dropped}",
            end="",
            flush=True,
        )

    wall_seconds = time.perf_counter() - started
    stats = capture.stop()
    while True:                                   # drain whatever is left
        chunk = capture.read(timeout=0.05)
        if chunk is None:
            break
        pcm.extend(chunk)
    print()

    audio_seconds = len(pcm) / 2.0 / TARGET_SAMPLE_RATE
    drift = abs(audio_seconds - wall_seconds) / max(wall_seconds, 1e-9)

    report.add("All chunks are exactly 6400 bytes", bad_sizes == 0,
               f"{bad_sizes} malformed chunk(s)")
    report.add("No chunk dropped by back-pressure", stats.chunks_dropped == 0,
               f"dropped={stats.chunks_dropped}")
    report.add("No PortAudio input overflow", stats.input_overflows == 0,
               f"overflows={stats.input_overflows}")
    report.add("Capture keeps up with real time",
               drift <= MAX_TIMING_DRIFT_RATIO,
               f"audio={audio_seconds:.2f}s vs wall={wall_seconds:.2f}s "
               f"(drift {drift * 100:.1f}%)")
    report.add("Chunk delivery is continuous",
               max_gap_ms < CHUNK_DURATION_MS * 5,
               f"largest gap between chunks: {max_gap_ms:.0f} ms")
    report.add("Consumer keeps the queue shallow", max_queue_depth < 25,
               f"peak queue depth: {max_queue_depth} chunks")

    print(
        f"\n  Device callbacks: {stats.callbacks}, "
        f"device frames: {stats.device_frames} "
        f"({stats.captured_seconds:.2f} s at {device.default_sample_rate} Hz)"
    )
    return bytes(pcm)


def analyse_signal(pcm: bytes, report: Report) -> None:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    if samples.size == 0:
        report.add("Captured audio is not empty", False, "0 samples")
        return

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    clipped = int(np.count_nonzero(np.abs(samples) >= 0.999))
    # Fraction of 20 ms frames that are essentially digital silence.
    frame = TARGET_SAMPLE_RATE * 20 // 1000
    frames = samples[: samples.size // frame * frame].reshape(-1, frame)
    silent_ratio = float(np.mean(np.max(np.abs(frames), axis=1) < 1e-4))

    print(
        f"\n  Peak: {dbfs(peak):.1f} dBFS | RMS: {dbfs(rms):.1f} dBFS | "
        f"silent frames: {silent_ratio * 100:.1f}% | clipped samples: {clipped}"
    )

    report.add("Signal is audible (peak level)", dbfs(peak) > MIN_PEAK_DBFS,
               f"peak {dbfs(peak):.1f} dBFS > {MIN_PEAK_DBFS} dBFS")
    report.add("Signal has energy (RMS level)", dbfs(rms) > MIN_RMS_DBFS,
               f"rms {dbfs(rms):.1f} dBFS > {MIN_RMS_DBFS} dBFS")
    report.add("Stream is not mostly digital silence", silent_ratio < 0.9,
               f"{silent_ratio * 100:.1f}% silent frames")
    report.add("Signal is not clipping", clipped < samples.size * 0.001,
               f"{clipped} clipped samples")


def save_wav(pcm: bytes) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"loopback_{datetime.now():%Y%m%d_%H%M%S}.wav"
    path.write_bytes(pcm16_to_wav_bytes(pcm))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="recording length in seconds (default: 10)")
    parser.add_argument("--device", default=None,
                        help="substring of the loopback device name to use")
    parser.add_argument("--list", action="store_true",
                        help="only list the available loopback devices")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 72)
    print("REAL TEST - WASAPI LOOPBACK AUDIO CAPTURE")
    print(f"Target format: {TARGET_SAMPLE_RATE} Hz / mono / 16-bit, "
          f"{CHUNK_DURATION_MS} ms chunks ({CHUNK_BYTES} bytes)")
    print("=" * 72)

    try:
        if args.list:
            print_devices()
            return 0

        print_devices()
        report = Report()
        print("\nChecks:")
        pcm = run_capture(args.seconds, args.device, report)
        analyse_signal(pcm, report)

        wav_path = save_wav(pcm)
        print(f"\n  Recording saved: {wav_path}")
        print("  Play it back and confirm you hear what was playing.")

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

    except AudioCaptureError as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
