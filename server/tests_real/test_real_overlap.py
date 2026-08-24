"""REAL TEST - the Overlap Resolver on real recorded voices.

MUST RUN ON: the GPU Server pod (or anywhere with the server requirements -
this stage needs no GPU).

``DESIGN.md`` section 3.4 asks for the louder voice to be favoured and the
quieter one squashed. This checks that on real speech rather than on tones,
using a mix built to a known level difference so there is a ground truth to
measure against.

Three signals are built from the recordings you already have:

``dominant``
    The speech recording as it is. The resolver must leave it essentially
    alone - if it damages a clean single voice, it is not worth having.

``bleed``
    The dominant voice, then a quieter voice after it. This is the far-end
    participant leaking through someone's speaker. The quiet half must come
    out measurably quieter, and the loud half must not.

``overlap``
    Both voices at once, which is the case the design names. Nothing here
    separates them - a gate cannot - so the check is only that the dominant
    voice survives intact. Whether the result is easier to transcribe is a
    question for the ASR stage and for your ears; the WAVs are written out.

Usage
-----
    python3.11 server/tests_real/test_real_overlap.py \\
        --speech recordings/meeting_speech.wav

    # with a genuinely different second voice
    python3.11 server/tests_real/test_real_overlap.py \\
        --speech recordings/meeting_speech.wav \\
        --quiet recordings/other_voice.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    CHANNELS,
    OVERLAP_GATE_BELOW_DB,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.overlap import (  # noqa: E402
    OverlapError,
    OverlapResolver,
    PedalboardProcessor,
    float_to_pcm,
    pcm_to_float,
    rms_dbfs,
    speaking_level_dbfs,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# How far under the dominant voice the second one is placed. Further down than
# the gate threshold, so the gate has a reason to act.
DEFAULT_BLEED_DB = 20.0
# A clean single voice must survive the stage essentially untouched.
MAX_DOMINANT_CHANGE_DB = 3.0
# ... and the quiet passage must be visibly attenuated.
MIN_BLEED_ATTENUATION_DB = 6.0


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


def read_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; expected 16000 Hz / mono / 16-bit"
            )
        return pcm_to_float(wav.readframes(wav.getnframes()))


def scale_to(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Rescale so the passage sits at a chosen level, whatever it started at."""
    level = rms_dbfs(samples)
    return samples * float(10.0 ** ((target_dbfs - level) / 20.0))


def write_wav(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(float_to_pcm(samples))
    return path


def second_voice(speech: np.ndarray, quiet_path: Optional[Path]) -> np.ndarray:
    """The interfering voice: a separate recording, or a different passage.

    With no second recording to hand, the far half of the same file stands in.
    It is the same speaker saying different words, which is a fair enough
    crosstalk signal for measuring a gate - the gate only ever sees level.
    """
    if quiet_path is not None:
        return read_samples(quiet_path)
    half = speech.size // 2
    return speech[half:]


# ---------------------------------------------------------------------------
# The three cases
# ---------------------------------------------------------------------------
def check_dominant(resolver: OverlapResolver, dominant: np.ndarray,
                   report: Report) -> np.ndarray:
    result = resolver.resolve(float_to_pcm(dominant))
    print(f"\n  dominant only: speaking {result.level_before_dbfs:.1f} -> "
          f"{result.level_after_dbfs:.1f} dBFS "
          f"(gate at {result.gate_threshold_dbfs:.1f} dBFS, "
          f"rms {rms_dbfs(dominant):.1f} -> "
          f"{rms_dbfs(pcm_to_float(result.pcm)):.1f})")
    report.add("A clean single voice is shaped, not skipped", result.shaped,
               result.reason)
    report.add(
        "A clean single voice comes through essentially unchanged",
        abs(result.gain_db) <= MAX_DOMINANT_CHANGE_DB,
        f"{result.gain_db:+.1f} dB, limit {MAX_DOMINANT_CHANGE_DB} dB",
    )
    report.add("The audio keeps its length",
               len(result.pcm) == len(float_to_pcm(dominant)),
               f"{len(result.pcm)} bytes")
    return pcm_to_float(result.pcm)


def check_bleed(resolver: OverlapResolver, loud: np.ndarray, quiet: np.ndarray,
                report: Report) -> np.ndarray:
    mixed = np.concatenate([loud, quiet])
    result = resolver.resolve(float_to_pcm(mixed))
    shaped = pcm_to_float(result.pcm)
    edge = loud.size

    # Measured on the speaking level, not the RMS. Gating the pauses inside the
    # loud half is the stage working correctly, and an RMS would read that as
    # damage to the voice.
    loud_before = speaking_level_dbfs(mixed[:edge])
    loud_after = speaking_level_dbfs(shaped[:edge])
    quiet_before = speaking_level_dbfs(mixed[edge:])
    quiet_after = speaking_level_dbfs(shaped[edge:])
    print(f"\n  bleed: gate at {result.gate_threshold_dbfs:.1f} dBFS")
    print(f"    loud half  speaking {loud_before:6.1f} -> {loud_after:6.1f} dBFS "
          f"({loud_after - loud_before:+.1f} dB), rms "
          f"{rms_dbfs(mixed[:edge]):.1f} -> {rms_dbfs(shaped[:edge]):.1f}")
    print(f"    quiet half speaking {quiet_before:6.1f} -> {quiet_after:6.1f} dBFS "
          f"({quiet_after - quiet_before:+.1f} dB), rms "
          f"{rms_dbfs(mixed[edge:]):.1f} -> {rms_dbfs(shaped[edge:]):.1f}")

    report.add(
        "The quieter voice is attenuated",
        quiet_after <= quiet_before - MIN_BLEED_ATTENUATION_DB,
        f"{quiet_after - quiet_before:+.1f} dB, wanted at most "
        f"-{MIN_BLEED_ATTENUATION_DB:.0f} dB",
    )
    report.add(
        "The dominant voice is not attenuated with it",
        loud_after >= loud_before - MAX_DOMINANT_CHANGE_DB,
        f"{loud_after - loud_before:+.1f} dB",
    )
    report.add(
        "The quieter voice ends up further below the dominant one",
        (loud_after - quiet_after) > (loud_before - quiet_before),
        f"separation {loud_before - quiet_before:.1f} dB -> "
        f"{loud_after - quiet_after:.1f} dB",
    )
    return shaped


def check_overlap(resolver: OverlapResolver, loud: np.ndarray,
                  quiet: np.ndarray, report: Report) -> np.ndarray:
    length = min(loud.size, quiet.size)
    mixed = loud[:length] + quiet[:length]
    result = resolver.resolve(float_to_pcm(mixed))
    shaped = pcm_to_float(result.pcm)
    print(f"\n  overlap: speaking {result.level_before_dbfs:.1f} -> "
          f"{result.level_after_dbfs:.1f} dBFS ({result.gain_db:+.1f} dB)")
    report.add(
        "Simultaneous voices are not destroyed by the stage",
        abs(result.gain_db) <= MAX_DOMINANT_CHANGE_DB * 2,
        f"{result.gain_db:+.1f} dB",
    )
    report.add("Every sample is finite",
               bool(np.all(np.isfinite(shaped))), "")
    report.add("No sample is clipped",
               float(np.max(np.abs(shaped))) < 0.999,
               f"peak {float(np.max(np.abs(shaped))):.3f}")
    return shaped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, required=True,
                        help="WAV of the dominant voice (16 kHz mono 16-bit)")
    parser.add_argument("--quiet", type=Path, default=None,
                        help="WAV of a second voice; defaults to the far half "
                             "of --speech")
    parser.add_argument("--bleed-db", type=float, default=DEFAULT_BLEED_DB,
                        help="how far under the dominant voice to place the "
                             f"second one (default: {DEFAULT_BLEED_DB:.0f} dB)")
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - OVERLAP RESOLVER (server pipeline, step 4)")
    print(f"gate opens {OVERLAP_GATE_BELOW_DB:.0f} dB under the utterance's "
          f"own level; second voice placed {args.bleed_db:.0f} dB down")
    print("=" * 72)

    report = Report()
    try:
        resolver = OverlapResolver(processor=PedalboardProcessor())
        report.add("The DSP backend loads", True, "pedalboard")

        dominant = read_samples(args.speech)
        interferer = second_voice(dominant, args.quiet)
        quiet = scale_to(interferer, rms_dbfs(dominant) - args.bleed_db)
        print(f"\n  dominant {rms_dbfs(dominant):.1f} dBFS from "
              f"{args.speech.name} ({dominant.size / SAMPLE_RATE:.1f} s)")
        print(f"  second voice placed at {rms_dbfs(quiet):.1f} dBFS "
              f"({'from ' + args.quiet.name if args.quiet else 'far half of the same file'})")

        print("\nChecks:")
        shaped_dominant = check_dominant(resolver, dominant, report)
        shaped_bleed = check_bleed(resolver, dominant, quiet, report)
        shaped_overlap = check_overlap(resolver, dominant, quiet, report)

        directory = OUTPUT_DIR / "overlap"
        write_wav(directory / "1_dominant_shaped.wav", shaped_dominant)
        write_wav(directory / "2_bleed_before.wav",
                  np.concatenate([dominant, quiet]))
        write_wav(directory / "2_bleed_shaped.wav", shaped_bleed)
        length = min(dominant.size, quiet.size)
        write_wav(directory / "3_overlap_before.wav",
                  dominant[:length] + quiet[:length])
        write_wav(directory / "3_overlap_shaped.wav", shaped_overlap)

        print(f"\n  Audio written to {directory}")
        print("  Listen to 1_dominant_shaped.wav first: the single voice must "
              "sound the same as it always did.")
        print("  Then compare 2_bleed_before/shaped: the second half should "
              "drop away without the first half changing.")

        print(f"\n  Resolver stats: {resolver.stats.seen} judged, "
              f"{resolver.stats.shaped} shaped, "
              f"mean change {resolver.stats.mean_reduction_db:+.1f} dB")

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

    except (OverlapError, ValueError, FileNotFoundError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
